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

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;
use smg_mesh::kv::CrdtNamespace;

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
/// All state lives in the mesh `configs()` CRDT namespace under typed key
/// prefixes; writes replicate to peers via the gossip loop.
pub struct MeshSyncManager {
    handler: Arc<MeshServerHandler>,
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
        Self { handler }
    }

    fn cfg(&self) -> Arc<CrdtNamespace> {
        self.handler.mesh_kv().configs()
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
            self.cfg().put(&format!("{WORKER_PREFIX}{worker_id}"), bytes);
        }
    }

    pub fn remove_worker_state(&self, worker_id: &str) {
        self.cfg().delete(&format!("{WORKER_PREFIX}{worker_id}"));
    }

    pub fn get_all_worker_states(&self) -> Vec<WorkerState> {
        let cfg = self.cfg();
        cfg.keys(WORKER_PREFIX)
            .iter()
            .filter_map(|k| cfg.get(k))
            .filter_map(|v| serde_json::from_slice(&v).ok())
            .collect()
    }

    pub fn get_worker_state(&self, worker_id: &str) -> Option<WorkerState> {
        self.cfg()
            .get(&format!("{WORKER_PREFIX}{worker_id}"))
            .and_then(|v| serde_json::from_slice(&v).ok())
    }

    // ---- policy state ----

    pub fn sync_policy_state(&self, model_id: String, _policy_name: String, config: Vec<u8>) {
        self.cfg().put(&format!("{POLICY_PREFIX}{model_id}"), config);
    }

    pub fn remove_policy_state(&self, model_id: &str) {
        self.cfg().delete(&format!("{POLICY_PREFIX}{model_id}"));
    }

    pub fn get_all_policy_states(&self) -> Vec<Value> {
        let cfg = self.cfg();
        cfg.keys(POLICY_PREFIX)
            .iter()
            .filter_map(|k| cfg.get(k))
            .map(|v| serde_json::from_slice(&v).unwrap_or(Value::Null))
            .collect()
    }

    pub fn get_policy_state(&self, model_id: &str) -> Option<Value> {
        self.cfg()
            .get(&format!("{POLICY_PREFIX}{model_id}"))
            .map(|v| serde_json::from_slice(&v).unwrap_or(Value::Null))
    }

    // ---- radix-cache tree state ----

    pub fn sync_tree_operation(&self, key: String, op: TreeOperation) -> Result<()> {
        let mut state = self.get_tree_state(&key).unwrap_or_default();
        state.operations.push(op);
        let bytes = serde_json::to_vec(&state)?;
        self.cfg().put(&format!("{TREE_PREFIX}{key}"), bytes);
        Ok(())
    }

    pub fn get_tree_state(&self, key: &str) -> Option<TreeState> {
        self.cfg()
            .get(&format!("{TREE_PREFIX}{key}"))
            .and_then(|v| serde_json::from_slice(&v).ok())
    }

    // ---- global rate limiting ----
    //
    // smg-mesh 1.4.1 dropped the built-in distributed rate-limit window.
    // These degrade to "never rate-limited"; rate limiting is still enforced
    // locally by the middleware token bucket.

    /// Returns `(is_exceeded, current_count, limit)`.
    pub fn check_global_rate_limit(&self) -> (bool, u64, u64) {
        (false, 0, 0)
    }

    pub fn update_rate_limit_membership(&self) {}

    pub fn get_rate_limit_value(&self, _key: &str) -> Option<i64> {
        Some(0)
    }
}

/// Per-window rate-limit reset task (was `smg_mesh::rate_limit_window`).
/// No-op under the local-only rate limiting model.
pub struct RateLimitWindow {
    _manager: Arc<MeshSyncManager>,
}

impl RateLimitWindow {
    pub fn new(manager: Arc<MeshSyncManager>, _reset_secs: u64) -> Self {
        Self { _manager: manager }
    }

    pub async fn start_reset_task(self) {
        // No periodic reset needed; counters are not mesh-replicated in 1.4.1.
    }
}
