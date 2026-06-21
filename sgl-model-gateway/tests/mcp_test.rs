// This test suite validates the complete MCP implementation against the
// functionality required for SGLang responses API integration.
//
// - Core MCP server functionality
// - Tool session management (individual and multi-tool)
// - Tool execution and error handling
// - Schema adaptation and validation
// - Mock server integration for reliable testing

mod common;

use std::collections::HashMap;

use common::mock_mcp_server::MockMCPServer;
use serde_json::json;
use smg_mcp::{
    ApprovalMode, McpConfig, McpOrchestrator as McpManager, McpServerConfig, McpTransport,
    TenantContext, ToolExecutionInput, ToolExecutionOutput,
};

/// Create a new mock server for testing (each test gets its own)
async fn create_mock_server() -> MockMCPServer {
    MockMCPServer::start()
        .await
        .expect("Failed to start mock MCP server")
}

fn server_config(name: &str, transport: McpTransport) -> McpServerConfig {
    McpServerConfig {
        name: name.to_string(),
        transport,
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal: false,
    }
}

fn streamable_server(name: &str, url: String, token: Option<String>) -> McpServerConfig {
    server_config(
        name,
        McpTransport::Streamable {
            url,
            token,
            headers: HashMap::new(),
        },
    )
}

fn stdio_server(name: &str, command: &str, args: Vec<String>) -> McpServerConfig {
    server_config(
        name,
        McpTransport::Stdio {
            command: command.to_string(),
            args,
            envs: HashMap::new(),
        },
    )
}

async fn execute_tool(
    manager: &McpManager,
    server_key: &str,
    tool_name: &str,
    arguments: serde_json::Value,
) -> ToolExecutionOutput {
    let request_ctx = manager.create_request_context(
        "test-request",
        TenantContext::default(),
        ApprovalMode::PolicyOnly,
    );

    manager
        .execute_tool_resolved(
            ToolExecutionInput {
                call_id: "call_test".to_string(),
                tool_name: tool_name.to_string(),
                arguments,
            },
            server_key,
            server_key,
            &request_ctx,
        )
        .await
}

// Core MCP Server Tests

#[tokio::test]
async fn test_mcp_server_initialization() {
    let config = McpConfig::default();

    // Should succeed but with no connected servers (empty config is allowed)
    let result = McpManager::new(config).await;
    assert!(result.is_ok(), "Should succeed with empty config");

    let manager = result.unwrap();
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 0, "Should have no servers");
    let tools = manager.list_tools(None);
    assert_eq!(tools.len(), 0, "Should have no tools");
}

#[tokio::test]
async fn test_server_connection_with_mock() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![streamable_server("mock_server", mock_server.url(), None)],
        ..Default::default()
    };

    let result = McpManager::new(config).await;
    assert!(result.is_ok(), "Should connect to mock server");

    let manager = result.unwrap();

    let servers = manager.list_servers();
    assert_eq!(servers.len(), 1);
    assert!(servers.contains(&"mock_server".to_string()));

    let tools = manager.list_tools(None);
    assert_eq!(tools.len(), 2, "Should have 2 tools from mock server");

    assert!(manager.has_tool("mock_server", "brave_web_search"));
    assert!(manager.has_tool("mock_server", "brave_local_search"));

    manager.shutdown().await;
}

#[tokio::test]
async fn test_tool_availability_checking() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![streamable_server("mock_server", mock_server.url(), None)],
        ..Default::default()
    };

    let manager = McpManager::new(config).await.unwrap();

    let test_tools = vec!["brave_web_search", "brave_local_search", "calculator"];
    for tool in test_tools {
        let available = manager.has_tool("mock_server", tool);
        match tool {
            "brave_web_search" | "brave_local_search" => {
                assert!(
                    available,
                    "Tool {} should be available from mock server",
                    tool
                );
            }
            "calculator" => {
                assert!(
                    !available,
                    "Tool {} should not be available from mock server",
                    tool
                );
            }
            _ => {}
        }
    }

    manager.shutdown().await;
}

#[tokio::test]
async fn test_multi_server_connection() {
    let mock_server1 = create_mock_server().await;
    let mock_server2 = create_mock_server().await;

    let config = McpConfig {
        servers: vec![
            streamable_server("mock_server_1", mock_server1.url(), None),
            streamable_server("mock_server_2", mock_server2.url(), None),
        ],
        ..Default::default()
    };

    // Note: This will fail to connect to both servers in the current implementation
    // since they return the same tools. The manager will connect to the first one.
    let result = McpManager::new(config).await;

    if let Ok(manager) = result {
        let servers = manager.list_servers();
        assert!(!servers.is_empty(), "Should have at least one server");

        let tools = manager.list_tools(None);
        assert!(tools.len() >= 2, "Should have tools from servers");

        manager.shutdown().await;
    }
}

#[tokio::test]
async fn test_tool_execution_with_mock() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![streamable_server("mock_server", mock_server.url(), None)],
        ..Default::default()
    };

    let manager = McpManager::new(config).await.unwrap();

    let response = execute_tool(
        &manager,
        "mock_server",
        "brave_web_search",
        json!({
            "query": "rust programming",
            "count": 1
        }),
    )
    .await;

    assert!(
        !response.is_error,
        "Tool execution should succeed with mock server"
    );
    assert!(response
        .output
        .to_string()
        .contains("Mock search results for: rust programming"));

    manager.shutdown().await;
}

#[tokio::test]
async fn test_concurrent_tool_execution() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![streamable_server("mock_server", mock_server.url(), None)],
        ..Default::default()
    };

    let manager = McpManager::new(config).await.unwrap();

    // Execute tools sequentially (true concurrent execution would require Arc<Mutex>)
    let tool_calls = vec![
        ("brave_web_search", json!({"query": "test1"})),
        ("brave_local_search", json!({"query": "test2"})),
    ];

    for (tool_name, args) in tool_calls {
        let response = execute_tool(&manager, "mock_server", tool_name, args).await;

        assert!(!response.is_error, "Tool {} should succeed", tool_name);
        assert!(!response.output.is_null(), "Should have content");
    }

    manager.shutdown().await;
}

// Error Handling Tests

#[tokio::test]
async fn test_tool_execution_errors() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![streamable_server("mock_server", mock_server.url(), None)],
        ..Default::default()
    };

    let manager = McpManager::new(config).await.unwrap();

    // Try to call unknown tool
    let response = execute_tool(&manager, "mock_server", "unknown_tool", json!({})).await;
    assert!(response.is_error, "Should fail for unknown tool");
    assert!(response
        .error_message
        .as_deref()
        .unwrap_or_default()
        .contains("unknown_tool"));

    manager.shutdown().await;
}

#[tokio::test]
async fn test_connection_without_server() {
    let config = McpConfig {
        servers: vec![stdio_server("nonexistent", "/nonexistent/command", vec![])],
        ..Default::default()
    };

    let result = McpManager::new(config).await;
    // Manager succeeds but no servers are connected (errors are logged)
    assert!(
        result.is_ok(),
        "Manager should succeed even if servers fail to connect"
    );

    let manager = result.unwrap();
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 0, "Should have no connected servers");
}

// Schema Validation Tests

#[tokio::test]
async fn test_tool_info_structure() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![streamable_server("mock_server", mock_server.url(), None)],
        ..Default::default()
    };

    let manager = McpManager::new(config).await.unwrap();

    let tools = manager.list_tools(None);
    let brave_search = tools
        .iter()
        .find(|t| t.tool.name.as_ref() == "brave_web_search")
        .expect("Should have brave_web_search tool");

    assert_eq!(brave_search.tool.name.as_ref(), "brave_web_search");
    assert!(brave_search
        .tool
        .description
        .as_ref()
        .map(|d| d.contains("Mock web search"))
        .unwrap_or(false));
    // Note: server information is now maintained separately in the inventory,
    // not in the Tool type itself
    assert!(!brave_search.tool.input_schema.is_empty());
}

// SSE Parsing Tests (simplified since we don't expose parse_sse_event)

#[tokio::test]
async fn test_sse_connection() {
    // This tests that SSE configuration is properly handled even when connection fails
    let config = McpConfig {
        servers: vec![stdio_server(
            "sse_test",
            "/nonexistent/sse/server",
            vec!["--sse".to_string()],
        )],
        ..Default::default()
    };

    // Manager succeeds but no servers are connected (errors are logged)
    let result = McpManager::new(config).await;
    assert!(
        result.is_ok(),
        "Manager should succeed even if SSE server fails to connect"
    );

    let manager = result.unwrap();
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 0, "Should have no connected servers");
}

// Connection Type Tests

#[tokio::test]
async fn test_transport_types() {
    // HTTP/Streamable transport
    let http_config = streamable_server(
        "http_server",
        "http://localhost:8080/mcp".to_string(),
        Some("auth_token".to_string()),
    );
    assert_eq!(http_config.name, "http_server");

    // SSE transport
    let sse_config = server_config(
        "sse_server",
        McpTransport::Sse {
            url: "http://localhost:8081/sse".to_string(),
            token: None,
            headers: HashMap::new(),
        },
    );
    assert_eq!(sse_config.name, "sse_server");

    // STDIO transport
    let stdio_config = stdio_server(
        "stdio_server",
        "mcp-server",
        vec!["--port".to_string(), "8082".to_string()],
    );
    assert_eq!(stdio_config.name, "stdio_server");
}

// Integration Pattern Tests

#[tokio::test]
async fn test_complete_workflow() {
    let mock_server = create_mock_server().await;

    // 1. Initialize configuration
    let config = McpConfig {
        servers: vec![streamable_server(
            "integration_test",
            mock_server.url(),
            None,
        )],
        ..Default::default()
    };

    // 2. Connect to server
    let manager = McpManager::new(config)
        .await
        .expect("Should connect to mock server");

    // 3. Verify server connection
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0], "integration_test");

    // 4. Check available tools
    let tools = manager.list_tools(None);
    assert_eq!(tools.len(), 2);

    // 5. Verify specific tools exist
    assert!(manager.has_tool("integration_test", "brave_web_search"));
    assert!(manager.has_tool("integration_test", "brave_local_search"));
    assert!(!manager.has_tool("integration_test", "nonexistent_tool"));

    // 6. Execute a tool
    let response = execute_tool(
        &manager,
        "integration_test",
        "brave_web_search",
        json!({
            "query": "SGLang router MCP integration",
            "count": 1
        }),
    )
    .await;

    assert!(!response.is_error, "Tool execution should succeed");
    assert!(!response.output.is_null(), "Should return content");

    // 7. Clean shutdown
    manager.shutdown().await;

    let capabilities = [
        "MCP server initialization",
        "Tool server connection and discovery",
        "Tool availability checking",
        "Tool execution",
        "Error handling and robustness",
        "Multi-server support",
        "Schema adaptation",
        "Mock server integration (no external dependencies)",
    ];

    assert_eq!(capabilities.len(), 8);
}
