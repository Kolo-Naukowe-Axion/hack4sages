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
        className={`inline-flex items-center gap-3 rounded-full font-display text-lg font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-cyan/20 ${
          disabled
            ? "cursor-not-allowed bg-surface px-10 py-4 text-muted opacity-50"
            : "bg-cyan px-10 py-4 text-white shadow-lg shadow-cyan/20 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-cyan/30"
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
