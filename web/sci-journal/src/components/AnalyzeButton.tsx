"use client";

import { Loader2 } from "lucide-react";

interface Props {
  disabled: boolean;
  loading: boolean;
  onClick: () => void;
}

export default function AnalyzeButton({ disabled, loading, onClick }: Props) {
  return (
    <div className="flex justify-center py-10">
      <button
        onClick={onClick}
        disabled={disabled || loading}
        className={`inline-flex items-center gap-2.5 border font-serif text-base font-semibold tracking-wide transition-all ${
          disabled
            ? "cursor-not-allowed border-border-light bg-surface px-8 py-3 text-muted opacity-50"
            : "border-accent bg-accent px-8 py-3 text-white shadow-sm hover:bg-accent-light"
        }`}
      >
        {loading && <Loader2 size={16} className="animate-spin" />}
        {loading ? "Analyzing Spectrum..." : "Run Biosignature Analysis"}
      </button>
    </div>
  );
}
