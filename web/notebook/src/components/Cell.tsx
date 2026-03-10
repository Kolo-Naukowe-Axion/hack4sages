import { ReactNode } from "react";

type CellType = "code" | "markdown" | "output";

interface CellProps {
  type: CellType;
  executionCount?: number | "*" | null;
  children: ReactNode;
  className?: string;
}

const borderColors: Record<CellType, string> = {
  code: "border-l-nb-blue",
  markdown: "border-l-nb-green",
  output: "border-l-nb-orange",
};

const promptLabels: Record<CellType, string> = {
  code: "In",
  markdown: "",
  output: "Out",
};

export default function Cell({ type, executionCount, children, className = "" }: CellProps) {
  const showPrompt = type !== "markdown";
  const promptText = executionCount !== undefined && executionCount !== null
    ? `${promptLabels[type]} [${executionCount}]:`
    : null;

  return (
    <div className={`bg-nb-cell border border-nb-border rounded ${borderColors[type]} border-l-[3px] ${className}`}>
      <div className="flex">
        {showPrompt && (
          <div className="shrink-0 w-[80px] pt-3 pr-2 text-right">
            {promptText && (
              <span className={`font-mono text-[12px] ${type === "output" ? "text-nb-orange" : "text-nb-prompt"} ${executionCount === "*" ? "cell-running" : ""}`}>
                {promptText}
              </span>
            )}
          </div>
        )}
        <div className={`flex-1 min-w-0 ${showPrompt ? "py-3 pr-4" : "p-4"}`}>
          {children}
        </div>
      </div>
    </div>
  );
}
