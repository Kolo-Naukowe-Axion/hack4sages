"use client";

import { Loader2, Scan } from "lucide-react";

interface Props {
  disabled: boolean;
  loading: boolean;
  onClick: () => void;
}

export default function AnalyzeButton({ disabled, loading, onClick }: Props) {
  return (
    <div className="flex justify-center py-12">
      <button
        onClick={onClick}
        disabled={disabled || loading}
        className={`inline-flex items-center gap-3 rounded-full font-display text-lg font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-cyan/50 ${
          disabled
            ? "cursor-not-allowed bg-muted/20 px-10 py-4 text-muted opacity-50"
            : "bg-cyan px-10 py-4 text-void shadow-[0_0_20px_rgba(0,229,255,0.3)] hover:scale-105 hover:shadow-[0_0_30px_rgba(0,229,255,0.5)]"
        }`}
      >
        {loading ? (
          <>
            <Loader2 size={20} className="animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <Scan size={20} />
            Analyze Atmosphere
          </>
        )}
      </button>
    </div>
  );
}
