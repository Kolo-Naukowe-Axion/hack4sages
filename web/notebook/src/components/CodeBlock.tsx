interface CodeBlockProps {
  lines: string[];
}

function highlightSyntax(line: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let remaining = line;
  let key = 0;

  const patterns: [RegExp, string][] = [
    [/^(#.*)$/, "text-nb-muted italic"],
    [/^(import|from|def|class|return|if|else|for|in|with|as|print)\b/, "text-[#af00db]"],
    [/\b(True|False|None)\b/, "text-[#0000ff]"],
    [/("""[\s\S]*?"""|'''[\s\S]*?'''|"[^"]*"|'[^']*')/, "text-[#a31515]"],
    [/(\d+\.?\d*)/, "text-[#098658]"],
    [/(=|==|\+|-|\*|\/|>|<|!=|:)/, "text-nb-text"],
  ];

  while (remaining.length > 0) {
    let matched = false;

    // Check for leading whitespace
    const wsMatch = remaining.match(/^(\s+)/);
    if (wsMatch) {
      parts.push(<span key={key++}>{wsMatch[1]}</span>);
      remaining = remaining.slice(wsMatch[1].length);
      continue;
    }

    // Check comment first (whole-line)
    if (remaining.trimStart().startsWith("#")) {
      const leadingWs = remaining.match(/^(\s*)/)?.[1] || "";
      parts.push(
        <span key={key++}>
          {leadingWs}
          <span className="text-nb-muted italic">{remaining.slice(leadingWs.length)}</span>
        </span>
      );
      remaining = "";
      matched = true;
    }

    if (matched) continue;

    // Check keywords
    for (const [pattern, cls] of patterns) {
      const m = remaining.match(pattern);
      if (m && m.index === 0) {
        parts.push(<span key={key++} className={cls}>{m[0]}</span>);
        remaining = remaining.slice(m[0].length);
        matched = true;
        break;
      }
    }

    if (!matched) {
      // Match identifiers or single chars
      const identMatch = remaining.match(/^[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*/);
      if (identMatch) {
        const ident = identMatch[0];
        // Color built-in functions
        const builtins = ["print", "len", "range", "type", "list", "dict", "set", "str", "int", "float"];
        const isBuiltin = builtins.includes(ident.split(".")[0]);
        parts.push(
          <span key={key++} className={isBuiltin ? "text-[#795e26]" : "text-nb-text"}>
            {ident}
          </span>
        );
        remaining = remaining.slice(ident.length);
      } else {
        parts.push(<span key={key++}>{remaining[0]}</span>);
        remaining = remaining.slice(1);
      }
    }
  }

  return <>{parts}</>;
}

export default function CodeBlock({ lines }: CodeBlockProps) {
  return (
    <div className="font-mono text-[13px] leading-relaxed bg-nb-code-bg rounded p-3 overflow-x-auto">
      {lines.map((line, i) => (
        <div key={i} className="whitespace-pre">
          {highlightSyntax(line)}
        </div>
      ))}
    </div>
  );
}
