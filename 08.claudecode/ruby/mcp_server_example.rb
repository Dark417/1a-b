# frozen_string_literal: true

# mcp_server_example.rb — a trivial MCP server (stdio, JSON-RPC 2.0).
#
# Bundled, dependency-free example server so mcp_client.rb can exercise a full
# MCP loop offline. Implements just enough of the spec to be real:
#   initialize / notifications/initialized / tools/list / tools/call.
# Newline-delimited JSON-RPC over stdin/stdout. Educational implementation of
# the public MCP spec (../docs/05.mcp.md).

require "json"

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
  {
    name: "add", title: "Add",
    description: "Add two integers and return the sum.",
    inputSchema: {
      type: "object",
      properties: { a: { type: "integer" }, b: { type: "integer" } },
      required: %w[a b]
    }
  },
  {
    name: "upper", title: "Uppercase",
    description: "Uppercase a string.",
    inputSchema: {
      type: "object",
      properties: { text: { type: "string" } },
      required: %w[text]
    }
  }
].freeze

def run_tool(name, args)
  case name
  when "add" then (args["a"].to_i + args["b"].to_i).to_s
  when "upper" then args["text"].to_s.upcase
  else raise "unknown tool #{name.inspect}"
  end
end

def handle(method, params)
  case method
  when "initialize"
    {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: "example-mcp-server", version: "0.1.0" }
    }
  when "tools/list"
    { tools: TOOLS }
  when "tools/call"
    { content: [{ type: "text", text: run_tool(params["name"], params["arguments"] || {}) }] }
  else
    raise "unknown method #{method.inspect}"
  end
end

$stdin.each_line do |line|
  line = line.strip
  next if line.empty?

  msg = JSON.parse(line)
  next unless msg.key?("id") # notifications expect no response

  response =
    begin
      { jsonrpc: "2.0", id: msg["id"], result: handle(msg["method"], msg["params"] || {}) }
    rescue StandardError => e
      { jsonrpc: "2.0", id: msg["id"], error: { code: -32_603, message: e.message } }
    end
  $stdout.puts(JSON.dump(response))
  $stdout.flush
end
