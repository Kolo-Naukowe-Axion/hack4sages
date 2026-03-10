"use client";

import { useState } from "react";
import { planets } from "@/data/planets";
import { getMockResults } from "@/data/mockResults";
import { StarField } from "@/components/StarField";
import { PlanetField } from "@/components/PlanetField";
import { PlanetDetailPanel } from "@/components/PlanetDetailPanel";
import { ResultsPanel } from "@/components/ResultsPanel";
import { ModelResult } from "@/types";
import { Telescope, X } from "lucide-react";

export default function ExplorerPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<ModelResult[] | null>(null);

  const selected = selectedId ? planets.find((p) => p.id === selectedId) : null;

  async function handleAnalyze() {
    if (!selectedId) return;
    setAnalyzing(true);
    setResults(null);
    try {
      const data = await getMockResults(selectedId);
      setResults(data.results);
    } finally {
      setAnalyzing(false);
    }
  }

  function handleSelect(id: string) {
    setSelectedId(id);
    setResults(null);
  }

  return (
    <div className="relative min-h-screen pt-16">
      <StarField />

      <div className="relative z-10 max-w-[1400px] mx-auto px-6 py-8">
        <div className="flex items-center gap-3 mb-6">
          <Telescope className="w-5 h-5 text-cyan" />
          <h1 className="text-2xl font-bold text-white">Planet Explorer</h1>
          <span className="text-sm text-white/30">
            Click a planet to select it, then analyze its atmosphere
          </span>
        </div>

        <div className="grid lg:grid-cols-[1fr_420px] gap-6">
          {/* Planet Field */}
          <div className="bg-space-800/30 border border-white/5 rounded-2xl overflow-hidden min-h-[500px] lg:min-h-[600px]">
            <PlanetField
              planets={planets}
              selectedId={selectedId}
              onSelect={handleSelect}
            />
          </div>

          {/* Detail Panel */}
          <div className="space-y-4 max-h-[calc(100vh-140px)] overflow-y-auto pr-1 scrollbar-thin">
            {selected ? (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-white/30 uppercase tracking-wider">
                    Selected Planet
                  </span>
                  <button
                    onClick={() => {
                      setSelectedId(null);
                      setResults(null);
                    }}
                    className="text-white/30 hover:text-white/60 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <PlanetDetailPanel
                  planet={selected}
                  onAnalyze={handleAnalyze}
                  analyzing={analyzing}
                />
                {results && <ResultsPanel results={results} planet={selected} />}
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-[400px] text-center">
                <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center mb-4">
                  <Telescope className="w-8 h-8 text-white/20" />
                </div>
                <p className="text-white/30 text-sm">
                  Select a planet from the field to view details
                </p>
                <p className="text-white/15 text-xs mt-1">
                  Hover over planets to see quick info
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="mt-6 flex flex-wrap items-center gap-6 text-xs text-white/30">
          <span className="font-medium text-white/50">Legend:</span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#ff6644]" /> Hot (&gt;450K)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#ff9944]" /> Warm (350-450K)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#44dd88]" /> Temperate (280-350K)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#44aaff]" /> Cool (220-280K)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#9966dd]" /> Cold (&lt;180K)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full border border-green/50" /> Habitable Zone
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-amber text-[6px] text-black font-bold flex items-center justify-center">J</span>
            JWST Data
          </span>
        </div>
      </div>
    </div>
  );
}
