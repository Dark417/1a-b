# frozen_string_literal: true

# mcp_client.rb — a minimal Model Context Protocol (MCP) client (Ruby port).
#
# MCP is the open standard Claude Code uses to connect to external tools/data via
# servers (../docs/05.mcp.md). Wire protocol: JSON-RPC 2.0 over a transport
# (stdio for local). Lifecycle: initialize -> notifications/initialized ->
# tools/list -> tools/call.
#
# This implements a small but real stdio JSON-RPC client and ships a bundled
# trivial server (mcp_server_example.rb) so the loop runs offline on the stdlib.
# Discovered MCP tools are adapted into Tool objects (namespaced mcp__<server>__).
# Educational implementation of the public MCP spec; not Anthropic's code.

require "json"
require "open3"
require_relative "tools"

module MiniAgent
  PROTOCOL_VERSION = "2025-06-18"

  # Speaks JSON-RPC 2.0 to an MCP server over its stdin/stdout pipes.
  class StdioMCPClient
    attr_reader :server_info, :capabilities

    def initialize(command)
      @command = command
      @id = 0
      @server_info = {}
      @capabilities = {}
    end

    def start
      @stdin, @stdout, @wait = Open3.popen2(*@command)
    end

    def stop
      @stdin&.close
      @wait&.kill rescue nil # rubocop:disable Style/RescueModifier
    end

    # Block helper: connect, initialize, yield, always stop.
    def self.open(command)
      client = new(command)
      client.start
      client.initialize_session
      yield client
    ensure
      client.stop
    end

    # --- lifecycle ---

    def initialize_session
      result = request("initialize", {
                         protocolVersion: PROTOCOL_VERSION,
                         capabilities: {},
                         clientInfo: { name: "mini-agent-mcp-client", version: "0.1.0" }
                       })
      @server_info = result["serverInfo"] || {}
      @capabilities = result["capabilities"] || {}
      notify("notifications/initialized")
    end

    # --- primitives ---

    def list_tools = request("tools/list")["tools"] || []

    def call_tool(name, arguments)
      result = request("tools/call", { name: name, arguments: arguments })
      (result["content"] || []).select { |b| b["type"] == "text" }.map { |b| b["text"] }.join("\n")
    end

    private

    def next_id
      @id += 1
    end

    def send_message(message)
      @stdin.puts(JSON.dump(message))
      @stdin.flush
    end

    def read_message
      line = @stdout.gets
      raise "MCP server closed the connection" unless line

      JSON.parse(line)
    end

    def request(method, params = {})
      rid = next_id
      send_message({ jsonrpc: "2.0", id: rid, method: method, params: params })
      loop do
        msg = read_message
        next unless msg["id"] == rid
        raise "MCP error: #{msg['error']}" if msg["error"]

        return msg["result"] || {}
      end
    end

    def notify(method, params = {})
      send_message({ jsonrpc: "2.0", method: method, params: params })
    end
  end

  # Adapt the server's MCP tools into Tool objects whose func calls back into the
  # server. Names namespaced mcp__<server>__<tool>, like Claude Code.
  def self.mcp_tools_as_agent_tools(client, prefix: "mcp__example__")
    client.list_tools.map do |spec|
      tool_name = spec["name"]
      Tool.new(
        name: prefix + tool_name,
        description: spec["description"].to_s,
        parameters: spec["inputSchema"] || { type: "object", properties: {} },
        func: lambda { |**kwargs|
          ToolResult.new(content: client.call_tool(tool_name, kwargs))
        },
        read_only: false
      )
    end
  end
end

# Self-test: spin up the bundled server, initialize, list, call.
if __FILE__ == $PROGRAM_NAME
  server = File.join(__dir__, "mcp_server_example.rb")
  MiniAgent::StdioMCPClient.open(["ruby", server]) do |client|
    puts "connected to: #{client.server_info}"
    puts "tools: #{client.list_tools.map { |t| t['name'] }}"
    puts "add(2,3) => #{client.call_tool('add', { a: 2, b: 3 })}"
    puts "upper('hi') => #{client.call_tool('upper', { text: 'hi' })}"
  end
end
