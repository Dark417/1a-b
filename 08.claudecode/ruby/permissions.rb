# frozen_string_literal: true

# permissions.rb — the permission gate (Ruby port).
#
# Before any tool runs, the request passes through a permission gate — an
# educational model of Claude Code's permission system (../docs/04.permissions.md):
#   * three decisions: allow / deny / ask; deny always wins over allow;
#   * rules are Tool(pattern) strings, e.g. "bash(git *)", with `*` wildcards;
#   * modes mirror Claude Code: default / acceptEdits / plan / bypass;
#   * read-only tools are auto-allowed;
#   * auto_approve answers every ask with "yes" for non-interactive demos.
# The gate sees only the tool name + args, like the real harness intercepting
# tool_use blocks before dispatch.

module MiniAgent
  module Decision
    ALLOW = :allow
    DENY  = :deny
    ASK   = :ask
  end

  module Mode
    DEFAULT      = :default       # ask before risky (mutating) tools
    ACCEPT_EDITS = :acceptEdits   # auto-allow file writes/edits, still ask for bash
    PLAN         = :plan          # read-only: block every mutating tool
    BYPASS       = :bypass        # allow everything (sandboxes/CI only)
  end

  READ_ONLY_TOOLS = %w[read_file glob grep list_dir].freeze
  EDIT_TOOLS = %w[write_file edit_file].freeze

  # Build the string a rule pattern matches against. For bash we expose the
  # command (so "bash(git *)" is meaningful); for file tools the path; else the
  # tool name. Mirrors Claude Code rules like Bash(npm run test *) / Read(./.env).
  def self.tool_signature(name, args)
    args = args.transform_keys(&:to_sym)
    return "bash(#{args[:command]})" if name == "bash"
    return "#{name}(#{args[:path]})" if args.key?(:path)
    return "#{name}(#{args[:pattern]})" if args.key?(:pattern)

    "#{name}()"
  end

  # An allow/deny/ask rule set + a mode, like a slice of settings.json.
  PermissionConfig = Struct.new(:allow, :deny, :ask, :mode, keyword_init: true) do
    def initialize(allow: [], deny: [], ask: [], mode: Mode::DEFAULT)
      super
    end
  end

  # fnmatch-based pattern test (File::FNM_PATHNAME off so `*` spans separators).
  def self.matches?(signature, patterns)
    patterns.any? { |p| File.fnmatch(p, signature) }
  end

  # Prompters: given a question, return true/false.
  AUTO_APPROVE = ->(_q) { true }
  AUTO_DENY = ->(_q) { false }
  CLI_PROMPTER = lambda do |question|
    print "#{question} [y/N] "
    %w[y yes].include?(($stdin.gets || "").strip.downcase)
  rescue StandardError
    false
  end

  # Decides whether a tool call may run, resolving `ask` via a prompter.
  class PermissionGate
    def initialize(config = PermissionConfig.new, prompter = CLI_PROMPTER)
      @config = config
      @prompter = prompter
    end

    # The raw decision (without prompting) for name(args).
    def decide(name, args, read_only: false) # rubocop:disable Metrics/CyclomaticComplexity,Metrics/PerceivedComplexity
      sig = MiniAgent.tool_signature(name, args)

      # 1. Explicit deny always wins.
      return Decision::DENY if MiniAgent.matches?(sig, @config.deny)

      # 2. Mode-level policy.
      mode = @config.mode
      return Decision::ALLOW if mode == Mode::BYPASS
      if mode == Mode::PLAN && !(read_only || READ_ONLY_TOOLS.include?(name))
        return Decision::DENY # plan mode: only read-only exploration
      end

      # 3. Read-only tools never need a gate.
      return Decision::ALLOW if read_only || READ_ONLY_TOOLS.include?(name)

      # 4. Explicit allow.
      return Decision::ALLOW if MiniAgent.matches?(sig, @config.allow)

      # 5. acceptEdits auto-allows file mutations.
      return Decision::ALLOW if mode == Mode::ACCEPT_EDITS && EDIT_TOOLS.include?(name)

      # 6. Explicit ask, else default to ask for mutating tools.
      Decision::ASK
    end

    # Resolve a decision into [allowed, reason], prompting if needed.
    def check(name, args, read_only: false)
      decision = decide(name, args, read_only: read_only)
      sig = MiniAgent.tool_signature(name, args)
      case decision
      when Decision::ALLOW then [true, "allowed: #{sig}"]
      when Decision::DENY then [false, "denied by policy: #{sig}"]
      else
        approved = @prompter.call("Allow #{sig}?")
        [approved, "#{approved ? 'approved by user' : 'rejected by user'}: #{sig}"]
      end
    end
  end
end
