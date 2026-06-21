//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::response::Response;
use data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage};
use serde_json::to_value;
use smg_mcp::McpOrchestrator as McpManager;
use tracing::{debug, error, warn};

use crate::{
    core::WorkerRegistry,
    protocols::{
        common::Tool,
        responses::{ResponseTool, ResponsesRequest, ResponsesResponse},
    },
    routers::{
        error,
        mcp_utils::{ensure_request_mcp_client, response_tool_as_function, response_tool_is_mcp},
        persistence_utils::persist_conversation_items,
    },
};

/// Ensure MCP connection succeeds if MCP tools are declared
///
/// Checks if request declares MCP tools, and if so, validates that
/// the MCP clients can be created and connected.
/// Returns Ok((has_mcp_tools, server_keys)) on success.
pub(crate) async fn ensure_mcp_connection(
    mcp_manager: &Arc<McpManager>,
    tools: Option<&[ResponseTool]>,
) -> Result<(bool, Vec<String>), Response> {
    let has_mcp_tools = tools
        .map(|t| t.iter().any(response_tool_is_mcp))
        .unwrap_or(false);

    if has_mcp_tools {
        if let Some(tools) = tools {
            match ensure_request_mcp_client(mcp_manager, tools).await {
                Some((_manager, server_keys)) => {
                    return Ok((true, server_keys));
                }
                None => {
                    error!(
                        function = "ensure_mcp_connection",
                        "Failed to connect to MCP servers"
                    );
                    return Err(error::failed_dependency(
                        "connect_mcp_server_failed",
                        "Failed to connect to MCP servers. Check server_url and authorization.",
                    ));
                }
            }
        }
    }

    Ok((false, Vec::new()))
}

/// Validate that workers are available for the requested model
pub(crate) fn validate_worker_availability(
    worker_registry: &Arc<WorkerRegistry>,
    model: &str,
) -> Option<Response> {
    let available_models = worker_registry.get_models();

    if !available_models.contains(&model.to_string()) {
        return Some(error::service_unavailable(
            "no_available_workers",
            format!(
                "No workers available for model '{}'. Available models: {}",
                model,
                available_models.join(", ")
            ),
        ));
    }

    None
}

/// Extract the function schemas from a set of ResponseTools.
///
/// Only `ResponseTool::Function` entries carry an inline schema and are
/// extracted. Discovered MCP tools reach here as `Function` entries too —
/// `convert_mcp_tools_to_response_tools()` rewrites them before this runs — so
/// they are picked up automatically. Raw `ResponseTool::Mcp` entries are server
/// references with no inline schema and are correctly skipped.
///
/// (Pre-migration this took an `include_mcp` flag because MCP tools were a
/// distinct variant that still carried a `function` payload; with the enum-based
/// `ResponseTool` that distinction is gone and the flag had no effect.)
pub(crate) fn extract_tools_from_response_tools(
    response_tools: Option<&[ResponseTool]>,
) -> Vec<Tool> {
    let Some(tools) = response_tools else {
        return Vec::new();
    };

    tools
        .iter()
        .filter_map(|rt| {
            let function = response_tool_as_function(rt)?;
            Some(Tool {
                tool_type: "function".to_string(),
                function: function.clone(),
            })
        })
        .collect()
}

/// Build a `reasoning` output item from analysis/reasoning text.
///
/// openai-protocol 1.8.x marks the `Reasoning` variants `#[non_exhaustive]` and
/// added the `encrypted_content` field, so they can no longer be built with a
/// struct literal from this crate. We construct them by deserializing a JSON
/// value instead; omitted fields (e.g. `encrypted_content`) fall back to their
/// serde defaults — matching the pre-migration behavior where the field did not
/// exist. Centralizing the shape here keeps the reasoning call sites in sync and
/// lets the round-trip test below guard against a protocol field/tag rename,
/// which would otherwise only surface at runtime.
///
/// `id_seed` is the unique suffix appended to the `reasoning_` id prefix; `T` is
/// the concrete item type (`ResponseOutputItem` or `ResponseInputOutputItem`).
pub(crate) fn build_reasoning_item<T: serde::de::DeserializeOwned>(
    id_seed: &str,
    text: &str,
) -> Result<T, serde_json::Error> {
    serde_json::from_value(serde_json::json!({
        "type": "reasoning",
        "id": format!("reasoning_{id_seed}"),
        "summary": [],
        "content": [{
            "type": "reasoning_text",
            "text": text,
        }],
        "status": "completed",
    }))
}

/// Persist response to storage if store=true
///
/// Common helper function to avoid duplication across sync and streaming paths
/// in both harmony and regular responses implementations.
pub(crate) async fn persist_response_if_needed(
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    original_request: &ResponsesRequest,
) {
    if !original_request.store.unwrap_or(true) {
        return;
    }

    if let Ok(response_json) = to_value(response) {
        if let Err(e) = persist_conversation_items(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response_json,
            original_request,
        )
        .await
        {
            warn!("Failed to persist response: {}", e);
        } else {
            debug!("Persisted response: {}", response.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::build_reasoning_item;
    use crate::protocols::responses::{ResponseInputOutputItem, ResponseOutputItem};

    /// The reasoning items are built by JSON deserialization (the protocol's
    /// `Reasoning` variants are `#[non_exhaustive]`), so the field/tag shape is
    /// no longer checked at compile time. This guards it: a protocol rename of
    /// `reasoning`/`reasoning_text` or a new required field on the variant fails
    /// here instead of silently breaking every reasoning response at runtime.
    #[test]
    fn build_reasoning_item_roundtrips_into_protocol_types() {
        let output: ResponseOutputItem =
            build_reasoning_item("resp_123", "thinking...").expect("output item deserializes");
        assert!(matches!(output, ResponseOutputItem::Reasoning { .. }));

        let input: ResponseInputOutputItem =
            build_reasoning_item("msg_123", "thinking...").expect("input item deserializes");
        assert!(matches!(input, ResponseInputOutputItem::Reasoning { .. }));
    }
}
