//! Compatibility facade for protocol crates.
//!
//! Most API types are re-exported from `openai-protocol`. A small set of
//! control-plane types was removed or moved in newer protocol releases, while
//! this crate still exposes those shapes through its own HTTP endpoints.

pub use openai_protocol::{
    builders, chat, classify, common, completion, embedding, event_types, generate, interactions,
    messages, model_card, model_type, models, multipart, parser, realtime_conversation,
    realtime_events, realtime_response, realtime_session, rerank, sampling_params, tokenize,
    transcription, validated, worker, UNKNOWN_MODEL_ID,
};

pub mod responses {
    pub use openai_protocol::responses::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, Deserialize)]
    pub struct ResponsesGetParams {
        #[serde(default)]
        pub include: Vec<String>,
        #[serde(default)]
        pub include_obfuscation: Option<bool>,
        #[serde(default)]
        pub starting_after: Option<i64>,
        #[serde(default)]
        pub stream: Option<bool>,
    }
}

pub mod worker_spec {
    use std::collections::HashMap;

    use axum::{
        http::StatusCode,
        response::{IntoResponse, Response},
        Json,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};

    use super::UNKNOWN_MODEL_ID;

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct WorkerConfigRequest {
        pub url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub api_key: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub priority: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cost: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub worker_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub bootstrap_port: Option<u16>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub runtime: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tokenizer_path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reasoning_parser: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_parser: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub chat_template: Option<String>,
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        pub labels: HashMap<String, String>,
        #[serde(default = "default_health_check_timeout")]
        pub health_check_timeout_secs: u64,
        #[serde(default = "default_health_check_interval")]
        pub health_check_interval_secs: u64,
        #[serde(default = "default_health_success_threshold")]
        pub health_success_threshold: u32,
        #[serde(default = "default_health_failure_threshold")]
        pub health_failure_threshold: u32,
        #[serde(default)]
        pub disable_health_check: bool,
        #[serde(default = "default_max_connection_attempts")]
        pub max_connection_attempts: u32,
        #[serde(default)]
        pub dp_aware: bool,
    }

    fn default_health_check_timeout() -> u64 {
        30
    }

    fn default_health_check_interval() -> u64 {
        60
    }

    fn default_health_success_threshold() -> u32 {
        2
    }

    fn default_health_failure_threshold() -> u32 {
        3
    }

    fn default_max_connection_attempts() -> u32 {
        20
    }

    #[derive(Debug, Clone, Serialize)]
    pub struct WorkerInfo {
        pub id: String,
        pub url: String,
        pub model_id: String,
        pub priority: u32,
        pub cost: f32,
        pub worker_type: String,
        pub is_healthy: bool,
        pub load: usize,
        pub connection_mode: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub runtime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tokenizer_path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reasoning_parser: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_parser: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub chat_template: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub bootstrap_port: Option<u16>,
        #[serde(skip_serializing_if = "HashMap::is_empty")]
        pub metadata: HashMap<String, String>,
        pub disable_health_check: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub job_status: Option<JobStatus>,
    }

    impl WorkerInfo {
        pub fn pending(worker_id: &str, url: String, job_status: Option<JobStatus>) -> Self {
            Self {
                id: worker_id.to_string(),
                url,
                model_id: UNKNOWN_MODEL_ID.to_string(),
                priority: 0,
                cost: 1.0,
                worker_type: UNKNOWN_MODEL_ID.to_string(),
                is_healthy: false,
                load: 0,
                connection_mode: UNKNOWN_MODEL_ID.to_string(),
                runtime_type: None,
                tokenizer_path: None,
                reasoning_parser: None,
                tool_parser: None,
                chat_template: None,
                bootstrap_port: None,
                metadata: HashMap::new(),
                disable_health_check: false,
                job_status,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JobStatus {
        pub job_type: String,
        pub worker_url: String,
        pub status: String,
        pub message: Option<String>,
        pub timestamp: u64,
    }

    impl JobStatus {
        pub fn pending(job_type: &str, worker_url: &str) -> Self {
            Self::new(job_type, worker_url, "pending", None)
        }

        pub fn processing(job_type: &str, worker_url: &str) -> Self {
            Self::new(job_type, worker_url, "processing", None)
        }

        pub fn failed(job_type: &str, worker_url: &str, error: String) -> Self {
            Self::new(job_type, worker_url, "failed", Some(error))
        }

        fn new(job_type: &str, worker_url: &str, status: &str, message: Option<String>) -> Self {
            Self {
                job_type: job_type.to_string(),
                worker_url: worker_url.to_string(),
                status: status.to_string(),
                message,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkerUpdateRequest {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub priority: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cost: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub labels: Option<HashMap<String, String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub api_key: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub health_check_timeout_secs: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub health_check_interval_secs: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub health_success_threshold: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub health_failure_threshold: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub disable_health_check: Option<bool>,
    }

    #[derive(Debug, Clone, Serialize)]
    pub struct WorkerErrorResponse {
        pub error: String,
        pub code: String,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct ServerInfo {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model_path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub priority: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cost: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub worker_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tokenizer_path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reasoning_parser: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_parser: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub chat_template: Option<String>,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct FlushCacheResult {
        pub successful: Vec<String>,
        pub failed: Vec<(String, String)>,
        pub total_workers: usize,
        pub http_workers: usize,
        pub message: String,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct WorkerLoadsResult {
        pub loads: Vec<WorkerLoadInfo>,
        pub total_workers: usize,
        pub successful: usize,
        pub failed: usize,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct WorkerLoadInfo {
        pub worker: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub worker_type: Option<String>,
        pub load: isize,
    }

    impl IntoResponse for FlushCacheResult {
        fn into_response(self) -> Response {
            let status = if self.failed.is_empty() {
                StatusCode::OK
            } else {
                StatusCode::PARTIAL_CONTENT
            };

            let mut body = json!({
                "status": if self.failed.is_empty() { "success" } else { "partial_success" },
                "message": self.message,
                "workers_flushed": self.successful.len(),
                "total_http_workers": self.http_workers,
                "total_workers": self.total_workers
            });

            if !self.failed.is_empty() {
                body["successful"] = json!(self.successful);
                body["failed"] = json!(self
                    .failed
                    .into_iter()
                    .map(|(url, err)| json!({"worker": url, "error": err}))
                    .collect::<Vec<_>>());
            }

            (status, Json(body)).into_response()
        }
    }

    impl IntoResponse for WorkerLoadsResult {
        fn into_response(self) -> Response {
            let loads: Vec<Value> = self
                .loads
                .iter()
                .map(|info| json!({"worker": &info.worker, "load": info.load}))
                .collect();
            Json(json!({"workers": loads})).into_response()
        }
    }
}
