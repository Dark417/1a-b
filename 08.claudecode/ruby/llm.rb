# frozen_string_literal: true

# llm.rb — backend abstraction for the mini coding agent (Ruby port).
#
# A *backend* turns the conversation so far (plus tool schemas) into the next
# assistant *step*: text + tool calls, or a final answer. This mirrors Claude
# Code's loop, which calls the Messages API with tools, inspects stop_reason,
# runs tool_use blocks, and feeds tool_result blocks back (see
# ../docs/02.architecture.md).
#
# Backends:
#   MockBackend     — deterministic, offline, default. Scripted planner so the
#                     demo runs with NO api key / network.
#   AnthropicBackend — optional; real Claude via the `anthropic` gem if present.
#   OllamaBackend   — optional; local open model via Ollama's HTTP API (stdlib).
#
# Only MockBackend is needed for `ruby demo.rb`. Original, clean-room
# educational code — not Anthropic's proprietary source.

module MiniAgent
  # A single request from the model to run a tool. `id` correlates the call with
  # its result (the API's tool_use_id); `name` selects a tool; `args` is input.
  ToolCall = Struct.new(:id, :name, :args, keyword_init: true)

  # One assistant turn produced by a backend.
  #   text       — prose to stream (may be empty)
  #   tool_calls — tools to run this turn; if present the loop runs them + recurs
  #   done       — final turn when true and no tool calls (== stop_reason end_turn)
  Step = Struct.new(:text, :tool_calls, :done, keyword_init: true) do
    def initialize(text: "", tool_calls: [], done: false)
      super
    end
  end

  # Base backend. `step` returns the next Step; `stream` yields text chunks for
  # incremental output (default: emit Step.text once, word by word).
  class Backend
    def name = "backend"

    def step(_messages, _tools)
      raise NotImplementedError
    end

    # Yields text fragments for the next step, then stores it for `last_step`.
    def stream(messages, tools)
      s = step(messages, tools)
      @last_step = s
      MiniAgent.word_chunks(s.text) { |chunk| yield chunk }
      s
    end

    def last_step
      raise "call stream before last_step" unless @last_step

      @last_step
    end
  end

  # Split text into small chunks to simulate token streaming.
  def self.word_chunks(text)
    return if text.nil? || text.empty?

    text.scan(/\s+|\S+/) { |part| yield part unless part.empty? }
  end

  # Distilled view of the conversation the mock reasons over.
  MockState = Struct.new(:task, :observations, :tool_calls_made, keyword_init: true)

  # Deterministic, offline backend driving a scripted agent demo.
  #
  # Default script implements "create a file and run it": list dir -> write file
  # -> run it -> summarize. Pass your own `script` (array of Steps or callables
  # taking a MockState) to drive other demos.
  class MockBackend < Backend
    def initialize(script: nil)
      super()
      @script = script
      @idx = 0
      @counter = 0
    end

    def name = "mock"

    def step(messages, _tools)
      state = distill(messages)
      return scripted_step(state) if @script

      default_plan(state)
    end

    private

    def next_id
      @counter += 1
      format("call_%03d", @counter)
    end

    def distill(messages)
      task = ""
      observations = []
      made = []
      messages.each do |m|
        content = m[:content]
        case m[:role]
        when "user"
          if content.is_a?(String)
            task = content if task.empty?
          elsif content.is_a?(Array)
            content.each do |b|
              observations << b[:content].to_s if b[:type] == "tool_result"
            end
          end
        when "assistant"
          next unless content.is_a?(Array)

          content.each { |b| made << b[:name] if b[:type] == "tool_use" }
        end
      end
      MockState.new(task: task, observations: observations, tool_calls_made: made)
    end

    def scripted_step(state)
      return Step.new(text: "(scripted plan complete)", done: true) if @idx >= @script.length

      item = @script[@idx]
      @idx += 1
      if item.respond_to?(:call)
        item.call(state) || Step.new(done: true)
      else
        # Step template: assign fresh ids to its tool calls.
        calls = item.tool_calls.map { |c| ToolCall.new(id: next_id, name: c.name, args: c.args) }
        Step.new(text: item.text, done: item.done, tool_calls: calls)
      end
    end

    # The built-in "write a file and run it" agent script. Each branch is gated
    # on what's already been done, so the planner is re-entrant.
    def default_plan(state)
      made = state.tool_calls_made

      unless made.include?("list_dir")
        return Step.new(
          text: "I'll start by looking at the working directory to orient myself.",
          tool_calls: [ToolCall.new(id: next_id, name: "list_dir", args: { path: "." })]
        )
      end

      unless made.include?("write_file")
        program = <<~PY
          print("hello from the mini coding agent")
          puts_equiv = 2 + 2
          print("2 + 2 =", puts_equiv)
        PY
        return Step.new(
          text: "The directory is empty. I'll create a small program `hello.py`.",
          tool_calls: [ToolCall.new(id: next_id, name: "write_file",
                                    args: { path: "hello.py", content: program })]
        )
      end

      unless made.include?("bash")
        return Step.new(
          text: "Now I'll run the program to verify it works.",
          tool_calls: [ToolCall.new(id: next_id, name: "bash",
                                    args: { command: "python3 hello.py" })]
        )
      end

      last = state.observations.last.to_s
      Step.new(
        text: "Done. I created `hello.py` and ran it successfully. " \
              "Its output was:\n#{last.strip}\nThe task is complete.",
        done: true
      )
    end
  end

  # Real Claude backend via the official `anthropic` gem (optional).
  # Used only when the gem is installed and ANTHROPIC_API_KEY is set. Defaults
  # to claude-opus-4-8 with adaptive thinking + streaming.
  class AnthropicBackend < Backend
    def initialize(model: "claude-opus-4-8", system: "")
      super()
      require "anthropic"
      raise "AnthropicBackend requires ANTHROPIC_API_KEY" unless ENV["ANTHROPIC_API_KEY"]

      @client = Anthropic::Client.new
      @model = model
      @system = system
    rescue LoadError => e
      raise "AnthropicBackend requires the `anthropic` gem: #{e.message}"
    end

    def name = "anthropic"

    def step(messages, tools) # rubocop:disable Metrics/MethodLength
      api_tools = tools.map do |t|
        { name: t[:name], description: t[:description].to_s, input_schema: t[:parameters] }
      end
      text = +""
      calls = []
      @client.messages.stream(
        model: @model, max_tokens: 8000, system: (@system.empty? ? nil : @system),
        thinking: { type: "adaptive" }, tools: api_tools, messages: messages
      ) do |event|
        if event.type == "content_block_delta" && event.delta.type == "text_delta"
          text << event.delta.text
        end
      end
      final = @client.messages.stream(model: @model, max_tokens: 8000, tools: api_tools,
                                      messages: messages).final_message
      final.content.each do |block|
        calls << ToolCall.new(id: block.id, name: block.name, args: block.input) if block.type == "tool_use"
      end
      Step.new(text: text, tool_calls: calls, done: final.stop_reason == "end_turn")
    end
  end

  # Local open-model backend via Ollama's HTTP API (optional, stdlib only).
  # Uses a JSON tool-call convention since many small local models lack native
  # tool calling.
  class OllamaBackend < Backend
    PROTOCOL = <<~TXT
      You are a coding agent with these tools (call by emitting a JSON object):
      %<tools>s
      To call tools reply with ONLY: {"tool_calls":[{"name":"<tool>","args":{...}}]}
      When finished reply with ONLY: {"done":true,"text":"<summary>"}
    TXT

    def initialize(model: "llama3.1", host: "http://localhost:11434")
      super()
      @model = model
      @host = host.sub(%r{/+\z}, "")
    end

    def name = "ollama"

    def step(messages, tools)
      require "net/http"
      require "json"
      tool_list = tools.map { |t| "- #{t[:name]}: #{t[:description]}" }.join("\n")
      system = format(PROTOCOL, tools: tool_list)
      payload = { model: @model, system: system, prompt: flatten(messages), stream: false }
      uri = URI("#{@host}/api/generate")
      res = Net::HTTP.post(uri, JSON.dump(payload), "Content-Type" => "application/json")
      parse(JSON.parse(res.body)["response"].to_s)
    end

    private

    def flatten(messages)
      lines = messages.map do |m|
        c = m[:content]
        c = JSON.dump(c) if c.is_a?(Array)
        "#{m[:role].upcase}: #{c}"
      end
      "#{lines.join("\n")}\nASSISTANT:"
    end

    def parse(response)
      m = response.match(/\{.*\}/m)
      return Step.new(text: response, done: true) unless m

      obj = JSON.parse(m[0])
      return Step.new(text: obj["text"].to_s, done: true) if obj["done"]

      calls = (obj["tool_calls"] || []).each_with_index.map do |c, i|
        ToolCall.new(id: "call_#{i}", name: c["name"], args: symbolize(c["args"] || {}))
      end
      Step.new(text: obj["text"].to_s, tool_calls: calls)
    rescue JSON::ParserError
      Step.new(text: response, done: true)
    end

    def symbolize(hash)
      hash.transform_keys(&:to_sym)
    end
  end

  # Pick the best available backend without failing: real Anthropic if a key +
  # gem are present, else Mock. The demo forces Mock explicitly.
  def self.auto_backend(system: "")
    if ENV["ANTHROPIC_API_KEY"]
      begin
        return AnthropicBackend.new(system: system)
      rescue StandardError
        # fall through
      end
    end
    MockBackend.new
  end
end
