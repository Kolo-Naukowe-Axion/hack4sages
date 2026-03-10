interface CodeBlockProps {
  lines: string[];
}

function highlightSyntax(line: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let remaining = line;
  let key = 0;

  const patterns: [RegExp, string][] = [
    [/^(#.*)$/, "text-[#5c6370] italic"],
    [/^(import|from|def|class|return|if|else|for|in|with|as|print)\b/, "text-[#c678dd]"],
    [/\b(True|False|None)\b/, "text-[#d19a66]"],
    [/("""[\s\S]*?"""|'''[\s\S]*?'''|"[^"]*"|'[^']*')/, "text-[#98c379]"],
    [/(\d+\.?\d*)/, "text-[#d19a66]"],
    [/(=|==|\+|-|\*|\/|>|<|!=|:)/, "text-[#abb2bf]"],
  ];

  while (remaining.length > 0) {
    let matched = false;

    const wsMatch = remaining.match(/^(\s+)/);
    if (wsMatch) {
      parts.push(<span key={key++}>{wsMatch[1]}</span>);
      remaining = remaining.slice(wsMatch[1].length);
      continue;
    }

    if (remaining.trimStart().startsWith("#")) {
      const leadingWs = remaining.match(/^(\s*)/)?.[1] || "";
      parts.push(
        <span key={key++}>
          {leadingWs}
          <span className="text-[#5c6370] italic">{remaining.slice(leadingWs.length)}</span>
        </span>
      );
      remaining = "";
      matched = true;
    }

    if (matched) continue;

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
      const identMatch = remaining.match(/^[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*/);
      if (identMatch) {
        const ident = identMatch[0];
        const builtins = ["print", "len", "range", "type", "list", "dict", "set", "str", "int", "float"];
        const isBuiltin = builtins.includes(ident.split(".")[0]);
        const isFunc = remaining[ident.length] === "(";
        parts.push(
          <span key={key++} className={isBuiltin ? "text-[#61afef]" : isFunc ? "text-[#61afef]" : "text-[#abb2bf]"}>
            {ident}
          </span>
        );
        remaining = remaining.slice(ident.length);
      } else {
        parts.push(<span key={key++} className="text-[#abb2bf]">{remaining[0]}</span>);
        remaining = remaining.slice(1);
      }
    }
  }

  return <>{parts}</>;
}

export default function CodeBlock({ lines }: CodeBlockProps) {
  return (
    <div className="font-mono text-[13px] leading-relaxed bg-nb-code-bg text-[#abb2bf] rounded-lg p-4 overflow-x-auto">
      {lines.map((line, i) => (
        <div key={i} className="whitespace-pre">
          {highlightSyntax(line)}
        </div>
      ))}
    </div>
  );
}
