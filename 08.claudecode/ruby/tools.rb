# frozen_string_literal: true

# tools.rb — the tool registry for the mini coding agent (Ruby port).
#
# Each tool is a callable plus a JSON-Schema-ish `parameters` block (the shape
# passed to the API's input_schema), registered in a ToolRegistry the agent loop
# dispatches against. Mirrors Claude Code's Read/Write/Edit/Bash/Glob/Grep
# design (see ../docs/03.tools.md):
#   * dedicated, typed, gateable file tools instead of raw bash;
#   * edit_file enforces a unique match (no ambiguous/corrupting edits);
#   * read-only tools are flagged parallel-safe;
#   * bash has a timeout.
# All paths resolve under a configurable root (a basic sandbox). Clean-room code.

require "open3"
require "pathname"

module MiniAgent
  # Result of running a tool. `is_error` maps to the API's tool_result flag.
  ToolResult = Struct.new(:content, :is_error, keyword_init: true) do
    def initialize(content:, is_error: false)
      super
    end
  end

  # A callable tool plus the metadata the model and harness need.
  Tool = Struct.new(:name, :description, :parameters, :func, :read_only, keyword_init: true) do
    def initialize(name:, description:, parameters:, func:, read_only: false)
      super
    end

    # The schema advertised to the backend (API tools[] entry shape).
    def schema
      { name: name, description: description, parameters: parameters, read_only: read_only }
    end
  end

  # Holds the tools, builds their schemas, and dispatches calls. Rooted at a
  # directory; all path args resolve relative to it and are confined inside it.
  class ToolRegistry
    def initialize(root = ".")
      @root = Pathname.new(root).expand_path
      @tools = {}
      register_builtins
    end

    attr_reader :root

    def register(tool) = @tools[tool.name] = tool
    def get(name) = @tools[name]
    def schemas = @tools.values.map(&:schema)
    def names = @tools.keys

    # Dispatch a call; tools should never crash the agent loop.
    def call(name, args)
      tool = @tools[name]
      return ToolResult.new(content: "Error: unknown tool #{name.inspect}", is_error: true) unless tool

      kwargs = args.transform_keys(&:to_sym)
      tool.func.call(**kwargs)
    rescue ArgumentError => e
      ToolResult.new(content: "Error: bad arguments for #{name}: #{e.message}", is_error: true)
    rescue StandardError => e
      ToolResult.new(content: "Error: #{e.class}: #{e.message}", is_error: true)
    end

    private

    # Resolve `path` under the root, rejecting escapes via `..`.
    def resolve(path)
      p = (@root + path).expand_path
      unless p == @root || p.to_s.start_with?("#{@root}/")
        raise ArgumentError, "path #{path.inspect} escapes the sandbox root"
      end

      p
    end

    def register_builtins # rubocop:disable Metrics/MethodLength,Metrics/AbcSize
      register(Tool.new(
                 name: "read_file",
                 description: "Read a UTF-8 text file and return its contents with line numbers.",
                 parameters: obj({ path: str("File path, relative to the workspace root.") }, %w[path]),
                 func: method(:read_file), read_only: true
               ))
      register(Tool.new(
                 name: "write_file",
                 description: "Write (create or overwrite) a UTF-8 text file. Creates parent dirs.",
                 parameters: obj({ path: str, content: str }, %w[path content]),
                 func: method(:write_file)
               ))
      register(Tool.new(
                 name: "edit_file",
                 description: "Replace an exact substring in a file. `old` must occur exactly once " \
                              "(prevents ambiguous/corrupting edits).",
                 parameters: obj({ path: str, old: str("Exact text to replace; must be unique."),
                                   new: str("Replacement text.") }, %w[path old new]),
                 func: method(:edit_file)
               ))
      register(Tool.new(
                 name: "bash",
                 description: "Run a shell command in the workspace root with a timeout. Returns stdout+stderr.",
                 parameters: obj({ command: str, timeout: int("Seconds (default 30).") }, %w[command]),
                 func: method(:bash)
               ))
      register(Tool.new(
                 name: "glob",
                 description: "Find files matching a glob pattern (e.g. '**/*.rb'). Returns matching paths.",
                 parameters: obj({ pattern: str }, %w[pattern]),
                 func: method(:glob), read_only: true
               ))
      register(Tool.new(
                 name: "grep",
                 description: "Search file contents for a regex. Returns matching 'path:lineno: line' rows.",
                 parameters: obj({ pattern: str, glob: str("Optional path filter (default '**/*').") }, %w[pattern]),
                 func: method(:grep), read_only: true
               ))
      register(Tool.new(
                 name: "list_dir",
                 description: "List the entries of a directory (default the workspace root).",
                 parameters: obj({ path: str }, []),
                 func: method(:list_dir), read_only: true
               ))
    end

    # --- tiny JSON-schema helpers ---
    def obj(props, required) = { type: "object", properties: props, required: required }
    def str(desc = nil) = desc ? { type: "string", description: desc } : { type: "string" }
    def int(desc = nil) = desc ? { type: "integer", description: desc } : { type: "integer" }

    # --- implementations ---

    def read_file(path:)
      p = resolve(path)
      return ToolResult.new(content: "Error: no such file #{path.inspect}", is_error: true) unless p.file?

      lines = p.read.split("\n", -1)
      lines.pop if lines.last == "" # drop trailing empty from final newline
      numbered = lines.each_with_index.map { |line, i| format("%4d\t%s", i + 1, line) }.join("\n")
      ToolResult.new(content: numbered.empty? ? "(empty file)" : numbered)
    end

    def write_file(path:, content:)
      p = resolve(path)
      p.dirname.mkpath
      p.write(content)
      n = content.empty? ? 0 : content.count("\n") + (content.end_with?("\n") ? 0 : 1)
      ToolResult.new(content: "Wrote #{content.bytesize} bytes (#{n} lines) to #{path}.")
    end

    def edit_file(path:, old:, new:)
      p = resolve(path)
      return ToolResult.new(content: "Error: no such file #{path.inspect}", is_error: true) unless p.file?

      text = p.read
      count = text.scan(old).length
      return ToolResult.new(content: "Error: `old` not found in #{path}.", is_error: true) if count.zero?
      if count > 1
        return ToolResult.new(content: "Error: `old` occurs #{count} times in #{path}; must be unique.",
                              is_error: true)
      end

      p.write(text.sub(old, new))
      ToolResult.new(content: "Edited #{path} (1 replacement).")
    end

    def bash(command:, timeout: 30)
      out, status = run_with_timeout(command, timeout)
      return ToolResult.new(content: "Error: command timed out after #{timeout}s.", is_error: true) if status == :timeout

      text = out.strip.empty? ? "(no output)" : out.strip
      prefix = status.zero? ? "" : "[exit #{status}]\n"
      ToolResult.new(content: prefix + text, is_error: !status.zero?)
    end

    def run_with_timeout(command, timeout)
      require "timeout"
      Open3.popen2e(command, chdir: @root.to_s) do |_stdin, out_err, wait_thr|
        begin
          output = Timeout.timeout(timeout) { out_err.read }
        rescue Timeout::Error
          Process.kill("KILL", wait_thr.pid)
          return ["", :timeout]
        end
        [output, wait_thr.value.exitstatus || 1]
      end
    end

    def glob(pattern:)
      matches = Dir.glob(pattern, base: @root.to_s).select { |m| (@root + m).file? }.sort
      ToolResult.new(content: matches.empty? ? "(no matches)" : matches.join("\n"))
    end

    def grep(pattern:, glob: "**/*")
      rx = Regexp.new(pattern)
      rows = []
      Dir.glob(glob, base: @root.to_s).sort.each do |rel|
        f = @root + rel
        next unless f.file?

        f.read.split("\n").each_with_index do |line, i|
          rows << "#{rel}:#{i + 1}: #{line.rstrip}" if rx.match?(line)
        end
      rescue StandardError
        next
      end
      ToolResult.new(content: rows.empty? ? "(no matches)" : rows.join("\n"))
    rescue RegexpError => e
      ToolResult.new(content: "Error: bad regex: #{e.message}", is_error: true)
    end

    def list_dir(path: ".")
      p = resolve(path)
      return ToolResult.new(content: "Error: not a directory #{path.inspect}", is_error: true) unless p.directory?

      entries = p.children.map { |e| e.directory? ? "#{e.basename}/" : e.basename.to_s }.sort
      ToolResult.new(content: entries.empty? ? "(empty directory)" : entries.join("\n"))
    end
  end
end

# Quick smoke test: `ruby tools.rb`
if __FILE__ == $PROGRAM_NAME
  require "tmpdir"
  Dir.mktmpdir do |d|
    reg = MiniAgent::ToolRegistry.new(d)
    puts reg.call("write_file", { path: "a.txt", content: "hi\nthere\n" }).content
    puts reg.call("read_file", { path: "a.txt" }).content
    puts reg.call("edit_file", { path: "a.txt", old: "hi", new: "hello" }).content
    puts reg.call("grep", { pattern: "hello" }).content
    puts reg.call("bash", { command: "echo ran" }).content
  end
end
