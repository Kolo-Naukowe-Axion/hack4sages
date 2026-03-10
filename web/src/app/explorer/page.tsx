"use client";

import { useState } from "react";
import { planets } from "@/data/planets";
import { getMockResults } from "@/data/mockResults";
import type { PlanetResults } from "@/types";
import PlanetTimeline from "@/components/PlanetTimeline";
import PlanetSummary from "@/components/PlanetSummary";
import AnalyzeButton from "@/components/AnalyzeButton";
import ResultsPanel from "@/components/ResultsPanel";
import BridgePanel from "@/components/BridgePanel";

export default function ExplorerPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [results, setResults] = useState<PlanetResults | null>(null);
  const [loading, setLoading] = useState(false);

  const selectedPlanet = planets.find((p) => p.id === selectedId) ?? null;

  const handleSelect = (id: string) => {
    setSelectedId(id);
    setResults(null);
    setLoading(false);
  };

  const handleAnalyze = async () => {
    if (!selectedId) return;
    setLoading(true);
    const data = await getMockResults(selectedId);
    setResults(data);
    setLoading(false);
  };

  return (
    <div className="mx-auto max-w-7xl px-6 py-12 lg:px-8">
      <h1 className="font-display text-4xl font-bold text-heading">
        Planet Explorer
      </h1>
      <p className="mt-3 max-w-2xl text-lg text-muted">
        Select an exoplanet to analyze its atmosphere for biosignatures
      </p>

      <div className="mt-10">
        <PlanetTimeline
          planets={planets}
          selectedId={selectedId}
          onSelect={handleSelect}
        />
      </div>

      {selectedPlanet && (
        <div key={selectedPlanet.id} className="mt-8">
          <PlanetSummary planet={selectedPlanet} />
        </div>
      )}

      <AnalyzeButton
        disabled={!selectedId}
        loading={loading}
        onClick={handleAnalyze}
      />

      {results && (
        <div key={`results-${results.planetId}`}>
          <ResultsPanel results={results} />
          <div className="pb-12">
            <BridgePanel />
          </div>
        </div>
      )}
    </div>
  );
}
