"use client";

import { Loader2, Scan } from "lucide-react";

interface Props {
  disabled: boolean;
  loading: boolean;
  onClick: () => void;
}

export default function AnalyzeButton({ disabled, loading, onClick }: Props) {
  return (
    <div className="flex justify-center py-14">
      <button
        onClick={onClick}
        disabled={disabled || loading}
        className={`inline-flex items-center gap-3 rounded-full font-display text-lg font-semibold transition-all duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan/30 focus-visible:ring-offset-2 ${
          disabled
            ? "cursor-not-allowed bg-surface px-10 py-4 text-muted/50"
            : "bg-heading px-10 py-4 text-white shadow-lg shadow-heading/15 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-heading/20 active:translate-y-0 active:shadow-md"
        }`}
      >
        <span className="inline-flex items-center gap-3">
          {loading ? (
            <Loader2 size={20} className="animate-spin" />
          ) : (
            <Scan size={20} />
          )}
          <span>{loading ? "Analyzing..." : "Analyze Atmosphere"}</span>
        </span>
      </button>
    </div>
  );
}
