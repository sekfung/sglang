//! Shared MCP utilities for routers.
//!
//! This module provides shared MCP-related functionality that can be
//! used across different router implementations (OpenAI, gRPC regular, gRPC harmony).

use std::sync::Arc;

use serde_json::{json, Value};
use smg_mcp::{
    ApprovalMode, McpOrchestrator as McpManager, McpServerConfig, McpTransport, TenantContext,
    Tool, ToolEntry, ToolExecutionInput, ToolExecutionOutput,
};
use tracing::warn;
use uuid::Uuid;

use crate::protocols::{
    common::Function,
    responses::{McpTool, ResponseTool},
};

// ============================================================================
// Constants
// ============================================================================

/// Default maximum tool loop iterations (safety limit).
///
/// Used as fallback when user doesn't specify `max_tool_calls`.
/// All routers use this same value.
pub const DEFAULT_MAX_ITERATIONS: usize = 10;

pub fn tool_entries_to_tools(entries: &[ToolEntry]) -> Vec<Tool> {
    entries.iter().map(|entry| entry.tool.clone()).collect()
}

pub fn response_tool_as_mcp(tool: &ResponseTool) -> Option<&McpTool> {
    match tool {
        ResponseTool::Mcp(mcp_tool) => Some(mcp_tool),
        _ => None,
    }
}

pub fn response_tool_as_function(tool: &ResponseTool) -> Option<&Function> {
    match tool {
        ResponseTool::Function(function_tool) => Some(&function_tool.function),
        _ => None,
    }
}

pub fn response_tool_is_mcp(tool: &ResponseTool) -> bool {
    matches!(tool, ResponseTool::Mcp(_))
}

pub async fn execute_mcp_tool(
    mcp_manager: &Arc<McpManager>,
    tool_name: &str,
    arguments: Value,
    call_id: impl Into<String>,
) -> Result<ToolExecutionOutput, String> {
    let call_id = call_id.into();
    let entry = mcp_manager
        .list_tools(None)
        .into_iter()
        .find(|entry| entry.tool.name.as_ref() == tool_name)
        .ok_or_else(|| format!("MCP tool '{}' not found", tool_name))?;

    let request_ctx = mcp_manager.create_request_context(
        format!("mcp_{}", Uuid::new_v4()),
        TenantContext::default(),
        ApprovalMode::PolicyOnly,
    );

    let output = mcp_manager
        .execute_tool_resolved(
            ToolExecutionInput {
                call_id,
                tool_name: entry.tool.name.to_string(),
                arguments,
            },
            entry.qualified_name.server_key(),
            entry.qualified_name.server_key(),
            &request_ctx,
        )
        .await;

    if output.is_error {
        Err(output.error_message.clone().unwrap_or_else(|| {
            output
                .output
                .get("error")
                .and_then(|v| v.as_str())
                .map(ToString::to_string)
                .unwrap_or_else(|| json!(output.output).to_string())
        }))
    } else {
        Ok(output)
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for MCP tool calling loops.
///
/// Provides a common structure for loop configuration across routers.
#[derive(Debug, Clone)]
pub struct McpLoopConfig {
    /// Maximum iterations as safety limit (default: DEFAULT_MAX_ITERATIONS).
    /// Prevents infinite loops when max_tool_calls is not set by user.
    pub max_iterations: usize,
    /// Server keys for filtering MCP tools.
    /// Contains keys for dynamic servers that were connected for this request.
    pub server_keys: Vec<String>,
}

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            server_keys: Vec::new(),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract MCP server label from request tools.
///
/// Searches for the first MCP tool in the tools array and returns its server_label.
/// Falls back to a default value if no MCP tool with server_label is found.
pub fn extract_server_label(tools: Option<&[ResponseTool]>, default_label: &str) -> String {
    tools
        .and_then(|tools| {
            tools
                .iter()
                .find_map(|tool| response_tool_as_mcp(tool).map(|tool| tool.server_label.clone()))
        })
        .unwrap_or_else(|| default_label.to_string())
}

// ============================================================================
// MCP Connection
// ============================================================================

/// Ensure MCP clients are connected for all request-level MCP tools.
///
/// This function extracts MCP server configurations from ALL request tools (server_url, authorization)
/// and ensures client connections are established via the connection pool.
///
/// Returns `Some((manager, server_keys))` if MCP tools were found and clients created,
/// `None` if no MCP tools with server_url were found.
pub async fn ensure_request_mcp_client(
    mcp_manager: &Arc<McpManager>,
    tools: &[ResponseTool],
) -> Option<(Arc<McpManager>, Vec<String>)> {
    let mut server_keys = Vec::new();
    let mut has_mcp_tools = false;

    // Process all MCP tools
    for tool in tools {
        if let Some(tool) = response_tool_as_mcp(tool).filter(|tool| tool.server_url.is_some()) {
            has_mcp_tools = true;
            let Some(server_url) = tool.server_url.as_ref().map(|s| s.trim().to_string()) else {
                continue;
            };

            // Validate URL scheme
            if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
                warn!(
                    "Ignoring MCP server_url with unsupported scheme: {}",
                    server_url
                );
                continue;
            }

            // Extract server label and auth token
            let name = tool.server_label.clone();
            let token = tool.authorization.clone();
            let headers = Default::default();

            // Determine transport type based on URL pattern
            let transport = if server_url.contains("/sse") {
                McpTransport::Sse {
                    url: server_url.clone(),
                    token,
                    headers,
                }
            } else {
                McpTransport::Streamable {
                    url: server_url.clone(),
                    token,
                    headers,
                }
            };

            // Create server config
            let server_config = McpServerConfig {
                name,
                transport,
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            };

            match mcp_manager.connect_dynamic_server(server_config).await {
                Ok(server_key) => {
                    // Track this server for filtering
                    if !server_keys.contains(&server_key) {
                        server_keys.push(server_key);
                    }
                }
                Err(err) => {
                    warn!(
                        "Failed to get/create MCP connection for {}: {}",
                        server_url, err
                    );
                    // Continue processing other tools
                }
            }
        }
    }

    if has_mcp_tools && !server_keys.is_empty() {
        Some((mcp_manager.clone(), server_keys))
    } else {
        None
    }
}
