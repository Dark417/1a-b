# frozen_string_literal: true

# agent.rb — the agent loop / REPL (Ruby port).
#
# The educational analog of Claude Code's main loop (../docs/02.architecture.md):
#   gather context -> ask model -> stream text
#     -> if it requested tools: gate each, run allowed ones, feed results back, recur
#     -> else: turn done
#
# Wires together the four subsystems: Backend (llm.rb), ToolRegistry (tools.rb),
# PermissionGate (permissions.rb), ContextManager (context.rb). Implements
# streaming-style incremental output and parallel read-only tool execution
# (read-only calls run on threads, mutating ones serially). Clean-room code.

require_relative "llm"
require_relative "tools"
require_relative "permissions"
require_relative "context"

module MiniAgent
  DEFAULT_SYSTEM = <<~SYS.strip
    You are a concise coding agent. Use the available tools to inspect and modify
    the workspace. Plan briefly, act with tools, verify your work, then stop.
    Prefer dedicated tools (read_file/write_file/edit_file) over bash for files.
  SYS

  # Structured record of one agent turn, useful for tests and transcripts.
  TurnLog = Struct.new(:text, :tool_calls, :results, :done, keyword_init: true)

  # Drives the gather->plan->act->observe loop to completion.
  class Agent
    attr_reader :turns, :context

    def initialize(backend: nil, tools: nil, gate: nil, context: nil,
                   system: DEFAULT_SYSTEM, out: $stdout, max_turns: 20, stream: true)
      @backend = backend || MockBackend.new
      @tools = tools || ToolRegistry.new(".")
      @gate = gate || PermissionGate.new
      @context = context || ContextManager.new(system: system)
      @out = out
      @max_turns = max_turns
      @stream = stream
      @turns = []
    end

    # Run the loop on `task`; return the final assistant text.
    def run(task)
      @context.add_user(task)
      final_text = ""

      @max_turns.times do
        step = ask_backend
        final_text = step.text
        @context.add_assistant(step.text, step.tool_calls)

        if step.tool_calls.empty?
          @turns << TurnLog.new(text: step.text, tool_calls: [], results: [], done: true)
          break
        end

        results = run_tools(step.tool_calls)
        @turns << TurnLog.new(text: step.text, tool_calls: step.tool_calls, results: results, done: step.done)

        @context.add_tool_results(
          step.tool_calls.zip(results).map { |call, (_, res)| [call.id, res.content, res.is_error] }
        )

        emit("\n[context compacted]\n") if @context.maybe_compact
      end

      final_text
    end

    private

    def emit(text)
      @out.write(text)
      @out.flush
    end

    def ask_backend
      messages = @context.to_api_messages
      schemas = @tools.schemas
      emit("\nassistant> ")
      if @stream
        @backend.stream(messages, schemas) { |chunk| emit(chunk) }
        step = @backend.last_step
      else
        step = @backend.step(messages, schemas)
        emit(step.text)
      end
      emit("\n")
      step
    end

    # Gate then run each call. Read-only calls run in parallel on threads.
    def run_tools(calls)
      results = {}
      parallel = []
      serial = []
      calls.each_with_index do |call, i|
        tool = @tools.get(call.name)
        (tool&.read_only ? parallel : serial) << [i, call]
      end

      threads = parallel.map do |i, call|
        Thread.new { results[i] = gate_and_run(call) }
      end
      threads.each(&:join)

      serial.each { |i, call| results[i] = gate_and_run(call) }

      (0...calls.length).map { |i| results[i] }
    end

    def gate_and_run(call)
      tool = @tools.get(call.name)
      read_only = tool ? tool.read_only : false
      allowed, reason = @gate.check(call.name, call.args, read_only: read_only)
      emit("  · #{call.name}(#{fmt_args(call.args)}) — #{reason}\n")
      unless allowed
        return [call.name, ToolResult.new(content: "Permission denied: #{reason}", is_error: true)]
      end

      result = @tools.call(call.name, call.args)
      preview = result.content.to_s.lines.first.to_s.strip
      emit("    #{preview[0, 100]}\n")
      [call.name, result]
    end

    def fmt_args(args)
      args.map { |k, v| "#{k}=#{v.to_s.gsub("\n", '\\n')[0, 40]}" }.join(", ")
    end
  end

  # A minimal interactive loop: read a task, run it, repeat.
  def self.repl(agent)
    puts "mini coding agent — type a task, or 'quit' to exit."
    loop do
      print "\nyou> "
      task = ($stdin.gets || "").strip
      break if task.empty? && $stdin.eof?
      break if %w[quit exit].include?(task.downcase)

      agent.run(task) unless task.empty?
    end
  end
end

if __FILE__ == $PROGRAM_NAME
  require "tmpdir"
  Dir.mktmpdir do |d|
    agent = MiniAgent::Agent.new(
      backend: MiniAgent::MockBackend.new,
      tools: MiniAgent::ToolRegistry.new(d),
      gate: MiniAgent::PermissionGate.new(MiniAgent::PermissionConfig.new, MiniAgent::AUTO_APPROVE)
    )
    MiniAgent.repl(agent)
  end
end
