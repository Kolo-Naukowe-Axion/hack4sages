import { ReactNode } from "react";

type CellType = "code" | "markdown" | "output";

interface CellProps {
  type: CellType;
  executionCount?: number | "*" | null;
  children: ReactNode;
  className?: string;
}

const accentColors: Record<CellType, string> = {
  code: "border-l-nb-blue",
  markdown: "border-l-nb-green",
  output: "border-l-nb-orange",
};

export default function Cell({ type, executionCount, children, className = "" }: CellProps) {
  const showPrompt = type !== "markdown";
  const promptNum = executionCount !== undefined && executionCount !== null
    ? `${executionCount}`
    : null;

  return (
    <div className={`bg-nb-cell rounded-lg shadow-sm border border-nb-border/60 ${accentColors[type]} border-l-2 ${className}`}>
      <div className="flex">
        {showPrompt && (
          <div className="shrink-0 w-[52px] pt-3.5 pr-1 text-right">
            {promptNum && (
              <span className={`inline-flex items-center justify-center font-mono text-[10px] w-6 h-5 rounded ${
                type === "output"
                  ? "bg-amber-50 text-nb-orange"
                  : "bg-indigo-50 text-nb-prompt"
              } ${executionCount === "*" ? "cell-running" : ""}`}>
                {promptNum}
              </span>
            )}
          </div>
        )}
        <div className={`flex-1 min-w-0 ${showPrompt ? "py-3.5 pr-4" : "p-4"}`}>
          {children}
        </div>
      </div>
    </div>
  );
}
