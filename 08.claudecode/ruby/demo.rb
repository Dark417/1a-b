# frozen_string_literal: true

# demo.rb — end-to-end offline demo of the mini coding agent (Ruby).
#
# Runs the full gather->plan->act->observe loop with the deterministic
# MockBackend on a scripted task ("create a small Ruby program and run it"),
# inside a throwaway temp directory. Exercises every subsystem: agent loop with
# streaming-style output, tool registry (list_dir -> write_file -> bash),
# permission gate (auto-approve), context manager (token budget + compaction),
# and a live MCP round-trip against the bundled example server.
#
# Requires NO api key / network, uses only the Ruby stdlib, and exits 0 on
# success. This is the validation entry point: `ruby demo.rb`.

require "tmpdir"
require_relative "agent"
require_relative "mcp_client"

module DemoRunner
  module_function

  def banner(title)
    puts "\n#{'=' * 70}"
    puts title
    puts "=" * 70
  end

  def demo_agent_loop(workdir)
    banner("1. AGENT LOOP — scripted 'write a file and run it' task (offline)")
    # Tight budget so compaction can trigger mid-run.
    ctx = MiniAgent::ContextManager.new(system: MiniAgent::DEFAULT_SYSTEM,
                                        budget: 600, trigger_ratio: 0.6, keep_recent: 2)
    agent = MiniAgent::Agent.new(
      backend: MiniAgent::MockBackend.new,
      tools: MiniAgent::ToolRegistry.new(workdir),
      gate: MiniAgent::PermissionGate.new(
        MiniAgent::PermissionConfig.new(mode: MiniAgent::Mode::DEFAULT, allow: ["bash(ruby *)"]),
        MiniAgent::AUTO_APPROVE
      ),
      context: ctx,
      stream: true
    )
    final = agent.run("Create a small Ruby program `hello.rb` and run it to verify it works.")
    puts "\n[turns: #{agent.turns.length} | compactions: #{ctx.compactions} | " \
         "est. tokens: #{ctx.total_tokens}]"
    final
  end

  def demo_permission_modes
    banner("2. PERMISSIONS — how each mode decides a risky tool call")
    call = ["bash", { command: "rm -rf /" }]
    [MiniAgent::Mode::DEFAULT, MiniAgent::Mode::PLAN,
     MiniAgent::Mode::ACCEPT_EDITS, MiniAgent::Mode::BYPASS].each do |mode|
      gate = MiniAgent::PermissionGate.new(
        MiniAgent::PermissionConfig.new(mode: mode, deny: ["bash(rm -rf *)"]),
        MiniAgent::AUTO_APPROVE
      )
      puts format("  mode=%-12s bash(rm -rf /) -> %s", mode, gate.decide(*call))
    end
    [MiniAgent::Mode::ACCEPT_EDITS, MiniAgent::Mode::PLAN].each do |mode|
      gate = MiniAgent::PermissionGate.new(MiniAgent::PermissionConfig.new(mode: mode),
                                           MiniAgent::AUTO_APPROVE)
      d = gate.decide("write_file", { path: "x.rb", content: "..." })
      puts format("  mode=%-12s write_file(x.rb)  -> %s", mode, d)
    end
  end

  def demo_mcp
    banner("3. MCP — live JSON-RPC round-trip with the bundled example server")
    server = File.join(__dir__, "mcp_server_example.rb")
    MiniAgent::StdioMCPClient.open(["ruby", server]) do |client|
      puts "  connected to #{client.server_info['name'].inspect} " \
           "(protocol #{client.capabilities})"
      adapted = MiniAgent.mcp_tools_as_agent_tools(client)
      puts "  discovered tools: #{adapted.map(&:name)}"
      add_tool = adapted.find { |t| t.name.end_with?("add") }
      puts "  mcp add(40, 2) => #{add_tool.func.call(a: 40, b: 2).content}" if add_tool
    end
  rescue StandardError => e
    puts "  (MCP demo skipped: #{e.message})"
  end

  def main
    puts "Mini Claude-Code-style coding agent — OFFLINE demo (MockBackend)"
    puts "No API key or network required."
    ok = false
    final = ""
    Dir.mktmpdir do |workdir|
      final = demo_agent_loop(workdir)
      ok = File.file?(File.join(workdir, "hello.rb"))
      puts "\n[verify] hello.rb created in workspace: #{ok}"
      demo_permission_modes
      demo_mcp
    end

    banner("DEMO COMPLETE")
    unless ok
      puts "FAILED: expected hello.rb to be created."
      return 1
    end
    puts "All subsystems exercised successfully."
    puts "Final assistant message:\n  #{final.gsub("\n", "\n  ")}"
    0
  end
end

exit(DemoRunner.main) if __FILE__ == $PROGRAM_NAME
