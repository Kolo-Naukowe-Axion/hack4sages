"use client";

import { useState, useRef, useCallback } from "react";
import { planets } from "@/data/planets";
import { getMockResults } from "@/data/mockResults";
import { PlanetField } from "@/components/PlanetField";
import { PlanetDetailPanel } from "@/components/PlanetDetailPanel";
import { AnalysisPipeline } from "@/components/AnalysisPipeline";
import { ResultsPanel } from "@/components/ResultsPanel";
import { ModelResult } from "@/types";

export default function ExplorerPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [pipelineActive, setPipelineActive] = useState(false);
  const [results, setResults] = useState<ModelResult[] | null>(null);
  const pendingResults = useRef<{ id: string; results: ModelResult[] } | null>(null);
  const analyzeIdRef = useRef<string | null>(null);

  const selected = selectedId ? planets.find((p) => p.id === selectedId) : null;

  async function handleAnalyze() {
    if (!selectedId) return;
    const currentId = selectedId;
    analyzeIdRef.current = currentId;
    setAnalyzing(true);
    setResults(null);
    setPipelineActive(true);
    pendingResults.current = null;

    try {
      const data = await getMockResults(currentId);
      if (analyzeIdRef.current === currentId) {
        pendingResults.current = { id: currentId, results: data.results };
      }
    } catch {
      pendingResults.current = null;
    }
  }

  const handlePipelineComplete = useCallback(() => {
    setPipelineActive(false);
    setAnalyzing(false);
    if (pendingResults.current && pendingResults.current.id === analyzeIdRef.current) {
      setResults(pendingResults.current.results);
    }
  }, []);

  function handleSelect(id: string) {
    setSelectedId(id);
    setResults(null);
    setPipelineActive(false);
    setAnalyzing(false);
    analyzeIdRef.current = null;
  }

  return (
    <div className="page-enter max-w-[1400px] mx-auto px-6 pt-20 pb-8">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-text">Planet Explorer</h1>
        <p className="text-sm text-muted mt-1">
          Click a planet to select it, then analyze its atmosphere
        </p>
      </div>

      {/* Planet Field */}
      <div className="w-full h-[60vh] bg-surface border border-border rounded-lg overflow-hidden">
        <PlanetField
          planets={planets}
          selectedId={selectedId}
          onSelect={handleSelect}
        />
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap items-center gap-5 text-xs text-muted">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-accent-red" /> Hot (&gt;350K)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-accent-amber" /> Warm (250-350K)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-accent-green" /> Temperate (200-250K)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-accent-blue" /> Cold (&lt;200K)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-accent-green" /> Habitable Zone
        </span>
        <span className="flex items-center gap-1.5">
          <span className="font-mono font-bold text-muted">J</span> JWST Data
        </span>
      </div>

      {/* Detail Panel */}
      {selected && (
        <div key={selected.id} className="mt-8 bg-surface border border-border rounded-lg p-6 fade-up">
          <PlanetDetailPanel
            planet={selected}
            onAnalyze={handleAnalyze}
            analyzing={analyzing}
          />
        </div>
      )}

      {/* Analysis Pipeline Animation */}
      {pipelineActive && (
        <div className="mt-6 fade-up">
          <AnalysisPipeline active={pipelineActive} onComplete={handlePipelineComplete} />
        </div>
      )}

      {/* Results */}
      {results && selected && (
        <div className="mt-6 fade-up">
          <ResultsPanel results={results} planet={selected} />
        </div>
      )}
    </div>
  );
}
