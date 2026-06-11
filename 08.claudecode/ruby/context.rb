# frozen_string_literal: true

# context.rb — conversation history + token budget / compaction (Ruby port).
#
# The Messages API is stateless: every turn resends the whole transcript. Long
# sessions grow until they threaten the context window, so real systems compact —
# summarize older turns, keep recent ones verbatim (../docs/02.architecture.md).
#
# ContextManager appends user/assistant/tool messages in the API's block shapes
# (tool_use / tool_result), estimates a token budget (~4 chars/token, offline),
# and compacts when over a threshold by replacing old messages with a
# deterministic summary (no LLM needed).

module MiniAgent
  # Very rough token estimate (~4 chars/token). Offline + approximate; real code
  # calls the count_tokens endpoint.
  def self.estimate_tokens(text)
    [1, text.to_s.length / 4].max
  end

  def self.message_tokens(message)
    content = message[:content] || ""
    return estimate_tokens(content) if content.is_a?(String)

    total = 0
    content.each do |block|
      case block[:type]
      when "text" then total += estimate_tokens(block[:text])
      when "tool_use" then total += estimate_tokens(block[:input].to_s)
      when "tool_result" then total += estimate_tokens(block[:content].to_s)
      end
    end
    total
  end

  # Holds the transcript and compacts it when over budget.
  class ContextManager
    attr_reader :messages, :compactions, :system

    def initialize(system: "", budget: 4000, trigger_ratio: 0.75, keep_recent: 4)
      @system = system
      @budget = budget
      @trigger_ratio = trigger_ratio
      @keep_recent = keep_recent
      @messages = []
      @compactions = 0
    end

    # --- appending ---

    def add_user(text)
      @messages << { role: "user", content: text }
    end

    # Add an assistant turn with optional tool_use blocks (ToolCall objects),
    # stored API-shaped so the next turn's tool_result blocks line up.
    def add_assistant(text, tool_calls = [])
      blocks = []
      blocks << { type: "text", text: text } unless text.nil? || text.empty?
      (tool_calls || []).each do |c|
        blocks << { type: "tool_use", id: c.id, name: c.name, input: c.args }
      end
      blocks = [{ type: "text", text: "" }] if blocks.empty?
      @messages << { role: "assistant", content: blocks }
    end

    # Add a user turn carrying tool_result blocks.
    # `results` is an array of [tool_use_id, content, is_error].
    def add_tool_results(results)
      blocks = results.map do |tid, content, is_error|
        { type: "tool_result", tool_use_id: tid, content: content, is_error: is_error }
      end
      @messages << { role: "user", content: blocks }
    end

    # --- budget / compaction ---

    def total_tokens
      MiniAgent.estimate_tokens(@system) + @messages.sum { |m| MiniAgent.message_tokens(m) }
    end

    # Compact if over the trigger threshold. Returns true if it compacted.
    def maybe_compact
      return false if total_tokens <= @trigger_ratio * @budget
      return false if @messages.length <= @keep_recent + 1

      compact!
      true
    end

    def to_api_messages = @messages.dup

    private

    def compact!
      head = @messages[0...-@keep_recent]
      tail = @messages[-@keep_recent..]
      @messages = [{ role: "user", content: summarize(head) }] + tail
      @compactions += 1
    end

    # Deterministic summary (no LLM): extract which tools ran and the first line
    # of each user message. Keeps the mechanism inspectable.
    def summarize(messages)
      actions = []
      messages.each do |m|
        content = m[:content]
        if m[:role] == "user" && content.is_a?(String)
          actions << "- user said: #{content.lines.first.to_s.strip[0, 80]}"
        elsif content.is_a?(Array)
          content.each do |block|
            case block[:type]
            when "tool_use"
              actions << "- ran tool: #{block[:name]}(#{block[:input]})"
            when "tool_result"
              first = block[:content].to_s.lines.first
              actions << "  -> #{first.strip[0, 80]}" if first
            end
          end
        end
      end
      body = actions.empty? ? "(no prior actions)" : actions.join("\n")
      "[COMPACTED SUMMARY of earlier conversation]\n#{body}"
    end
  end
end
