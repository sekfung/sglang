//! Compatibility layer bridging the gateway's mesh code (written for
//! smg-mesh 1.0) to smg-mesh 1.4.1.
//!
//! smg-mesh 1.4.1 restructured its public API:
//! - `MeshSyncManager`, `StateStores`, `tree_ops`, `RateLimitWindow` removed
//! - `OptionalMeshSyncManager` alias removed
//! - state sync now goes through the CRDT-based `kv::MeshKV` / `CrdtNamespace`
//! - `service`/`partition`/`metrics`/`stores`/`sync` modules made private
//!   (their public types are re-exported at the crate root)
//! - `MeshServerConfig` split `self_addr` into `bind_addr` + `advertise_addr`
//!
//! This module reimplements the old high-level `MeshSyncManager` API on top of
//! the new `MeshKV` CRDT namespace so the gateway's existing call sites compile
//! and keep their semantics (worker/policy/tree state replication across nodes).

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use serde::Deserialize;
use serde_json::Value;
use smg_mesh::{decode_epoch_count, encode_epoch_count, kv::CrdtNamespace, MergeStrategy};
// Re-export crate-root types that the gateway previously reached through the
// now-private `service` / `partition` / `metrics` submodules.
pub use smg_mesh::{
    init_mesh_metrics, ClusterState, MeshServerBuilder, MeshServerConfig, MeshServerHandler,
    PartitionDetector, WorkerState,
};

/// `Option<Arc<MeshSyncManager>>` — kept for source compatibility.
pub type OptionalMeshSyncManager = Option<Arc<MeshSyncManager>>;

const WORKER_PREFIX: &str = "worker:";
const POLICY_PREFIX: &str = "policy:";
const TREE_PREFIX: &str = "tree:";
const RATE_LIMIT_PREFIX: &str = "rl:";
const GLOBAL_RATE_LIMIT_KEY: &str = "config:global_rate_limit";
const GLOBAL_RATE_LIMIT_COUNTER_KEY: &str = "global_rate_limit_counter";

#[derive(Debug, Default, Deserialize)]
struct RateLimitConfig {
    limit_per_second: u64,
}

/// CRDT-replicated radix-cache tree operations (was `smg_mesh::tree_ops`).
pub mod tree_ops {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct TreeInsertOp {
        pub text: String,
        pub tenant: String,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct TreeRemoveOp {
        pub tenant: String,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub enum TreeOperation {
        Insert(TreeInsertOp),
        Remove(TreeRemoveOp),
    }

    /// Accumulated tree operations for a model, replicated via the mesh.
    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    pub struct TreeState {
        pub operations: Vec<TreeOperation>,
    }
}

use tree_ops::{TreeOperation, TreeState};

/// High-level mesh state synchronizer, reimplemented over `MeshKV` CRDT kv.
///
/// State lives in typed CRDT namespaces that match the old high-level stores;
/// writes replicate to peers via the gossip loop.
pub struct MeshSyncManager {
    handler: Arc<MeshServerHandler>,
    workers: Arc<CrdtNamespace>,
    policies: Arc<CrdtNamespace>,
    trees: Arc<CrdtNamespace>,
    rate_limits: Arc<CrdtNamespace>,
}

impl std::fmt::Debug for MeshSyncManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshSyncManager")
            .field("self_name", &self.handler.self_name)
            .finish()
    }
}

impl MeshSyncManager {
    pub fn new(handler: Arc<MeshServerHandler>) -> Self {
        let mesh_kv = handler.mesh_kv();
        let workers = mesh_kv.configure_crdt_prefix(WORKER_PREFIX, MergeStrategy::LastWriterWins);
        let policies = mesh_kv.configure_crdt_prefix(POLICY_PREFIX, MergeStrategy::LastWriterWins);
        let trees = mesh_kv.configure_crdt_prefix(TREE_PREFIX, MergeStrategy::LastWriterWins);
        let rate_limits =
            mesh_kv.configure_crdt_prefix(RATE_LIMIT_PREFIX, MergeStrategy::EpochMaxWins);
        Self {
            handler,
            workers,
            policies,
            trees,
            rate_limits,
        }
    }

    fn cfg(&self) -> Arc<CrdtNamespace> {
        self.handler.mesh_kv().configs()
    }

    fn current_epoch() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default()
    }

    fn rate_limit_key(&self, key: &str) -> String {
        format!("{RATE_LIMIT_PREFIX}{key}:{}", self.handler.self_name)
    }

    // ---- worker state ----

    pub fn sync_worker_state(
        &self,
        worker_id: String,
        model_id: String,
        url: String,
        health: bool,
        load: f64,
    ) {
        let ws = WorkerState {
            worker_id: worker_id.clone(),
            model_id,
            url,
            health,
            load,
            version: 0,
            spec: Vec::new(),
        };
        if let Ok(bytes) = serde_json::to_vec(&ws) {
            self.workers
                .put(&format!("{WORKER_PREFIX}{worker_id}"), bytes);
        }
    }

    pub fn remove_worker_state(&self, worker_id: &str) {
        self.workers.delete(&format!("{WORKER_PREFIX}{worker_id}"));
    }

    pub fn get_all_worker_states(&self) -> Vec<WorkerState> {
        self.workers
            .keys("")
            .iter()
            .filter_map(|k| self.workers.get(k))
            .filter_map(|v| serde_json::from_slice(&v).ok())
            .collect()
    }

    pub fn get_worker_state(&self, worker_id: &str) -> Option<WorkerState> {
        self.workers
            .get(&format!("{WORKER_PREFIX}{worker_id}"))
            .and_then(|v| serde_json::from_slice(&v).ok())
    }

    // ---- policy state ----

    pub fn sync_policy_state(&self, model_id: String, _policy_name: String, config: Vec<u8>) {
        self.policies
            .put(&format!("{POLICY_PREFIX}{model_id}"), config);
    }

    pub fn remove_policy_state(&self, model_id: &str) {
        self.policies.delete(&format!("{POLICY_PREFIX}{model_id}"));
    }

    pub fn get_all_policy_states(&self) -> Vec<Value> {
        self.policies
            .keys("")
            .iter()
            .filter_map(|k| self.policies.get(k))
            .map(|v| serde_json::from_slice(&v).unwrap_or(Value::Null))
            .collect()
    }

    pub fn get_policy_state(&self, model_id: &str) -> Option<Value> {
        self.policies
            .get(&format!("{POLICY_PREFIX}{model_id}"))
            .map(|v| serde_json::from_slice(&v).unwrap_or(Value::Null))
    }

    // ---- radix-cache tree state ----

    pub fn sync_tree_operation(&self, key: String, op: TreeOperation) -> Result<()> {
        let mut state = self.get_tree_state(&key).unwrap_or_default();
        state.operations.push(op);
        let bytes = serde_json::to_vec(&state)?;
        self.trees.put(&format!("{TREE_PREFIX}{key}"), bytes);
        Ok(())
    }

    pub fn get_tree_state(&self, key: &str) -> Option<TreeState> {
        self.trees
            .get(&format!("{TREE_PREFIX}{key}"))
            .and_then(|v| serde_json::from_slice(&v).ok())
    }

    // ---- global rate limiting ----

    /// Returns `(is_exceeded, current_count, limit)`.
    pub fn check_global_rate_limit(&self) -> (bool, u64, u64) {
        let limit = self
            .cfg()
            .get(GLOBAL_RATE_LIMIT_KEY)
            .and_then(|v| serde_json::from_slice::<RateLimitConfig>(&v).ok())
            .map(|c| c.limit_per_second)
            .unwrap_or_default();
        if limit == 0 {
            return (false, 0, 0);
        }

        let epoch = Self::current_epoch();
        let local_key = self.rate_limit_key(GLOBAL_RATE_LIMIT_COUNTER_KEY);
        let local_count = self
            .rate_limits
            .get(&local_key)
            .and_then(|v| decode_epoch_count(&v))
            .filter(|v| v.epoch == epoch)
            .map(|v| v.count.saturating_add(1))
            .unwrap_or(1);
        self.rate_limits
            .put(&local_key, encode_epoch_count(epoch, local_count).to_vec());

        let current_count = self.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
        let current_count_u64 = current_count.unwrap_or(0).max(0) as u64;
        (current_count_u64 > limit, current_count_u64, limit)
    }

    pub fn update_rate_limit_membership(&self) {}

    pub fn get_rate_limit_value(&self, key: &str) -> Option<i64> {
        let epoch = Self::current_epoch();
        let sub_prefix = format!("{key}:");
        let count = self
            .rate_limits
            .keys(&sub_prefix)
            .iter()
            .filter_map(|k| self.rate_limits.get(k))
            .filter_map(|v| decode_epoch_count(&v))
            .filter(|v| v.epoch == epoch)
            .map(|v| v.count)
            .sum();
        Some(count)
    }
}

/// Per-window rate-limit reset task (was `smg_mesh::rate_limit_window`).
/// No-op: counters are epoch-scoped in the `rl:` CRDT namespace.
pub struct RateLimitWindow {
    _manager: Arc<MeshSyncManager>,
}

impl RateLimitWindow {
    pub fn new(manager: Arc<MeshSyncManager>, _reset_secs: u64) -> Self {
        Self { _manager: manager }
    }

    pub async fn start_reset_task(self) {
        // No periodic reset needed; counters roll over by epoch.
    }
}

#[cfg(test)]
mod tests {
    use std::{net::SocketAddr, sync::Arc};

    use serde_json::json;

    use super::{
        tree_ops::{TreeInsertOp, TreeOperation},
        MeshServerBuilder, MeshSyncManager,
    };

    fn test_manager() -> Arc<MeshSyncManager> {
        let addr = SocketAddr::from(([127, 0, 0, 1], 0));
        let (_server, handler) =
            MeshServerBuilder::new("test-node".to_string(), addr, addr, None).build();
        Arc::new(MeshSyncManager::new(Arc::new(handler)))
    }

    #[test]
    fn mesh_state_round_trips_without_namespace_panics() {
        let manager = test_manager();

        manager.sync_worker_state(
            "worker-a".to_string(),
            "model-a".to_string(),
            "http://worker-a".to_string(),
            true,
            3.0,
        );
        let worker = manager.get_worker_state("worker-a").unwrap();
        assert_eq!(worker.worker_id, "worker-a");
        assert_eq!(worker.model_id, "model-a");
        assert_eq!(manager.get_all_worker_states().len(), 1);

        let policy = serde_json::to_vec("cache_aware").unwrap();
        manager.sync_policy_state("model-a".to_string(), "cache_aware".to_string(), policy);
        assert_eq!(
            manager.get_policy_state("model-a").unwrap(),
            json!("cache_aware")
        );

        manager
            .sync_tree_operation(
                "regular::model-a".to_string(),
                TreeOperation::Insert(TreeInsertOp {
                    text: "hello".to_string(),
                    tenant: "http://worker-a".to_string(),
                }),
            )
            .unwrap();
        let tree = manager.get_tree_state("regular::model-a").unwrap();
        assert_eq!(tree.operations.len(), 1);
    }

    #[test]
    fn global_rate_limit_rejects_after_configured_limit() {
        let manager = test_manager();
        let config = serde_json::to_vec(&json!({ "limit_per_second": 2 })).unwrap();
        manager
            .handler
            .mesh_kv()
            .configs()
            .put("config:global_rate_limit", config);

        assert_eq!(manager.check_global_rate_limit(), (false, 1, 2));
        assert_eq!(manager.check_global_rate_limit(), (false, 2, 2));
        assert_eq!(manager.check_global_rate_limit(), (true, 3, 2));
    }
}
