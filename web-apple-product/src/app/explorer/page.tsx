"use client";

import { useRef, useState } from "react";
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
  const requestRef = useRef(0);

  const selectedPlanet = planets.find((p) => p.id === selectedId) ?? null;

  const handleSelect = (id: string) => {
    setSelectedId(id);
    setResults(null);
    setLoading(false);
    requestRef.current++;
  };

  const handleAnalyze = async () => {
    if (!selectedId) return;
    const currentRequest = ++requestRef.current;
    setLoading(true);
    const data = await getMockResults(selectedId);
    if (currentRequest !== requestRef.current) return;
    setResults(data);
    setLoading(false);
  };

  return (
    <div className="mx-auto max-w-7xl px-6 py-16 lg:px-8">
      <h1 className="font-display text-4xl font-medium tracking-tight text-heading sm:text-5xl">
        Planet Explorer
      </h1>
      <p className="mt-3 max-w-lg text-base text-muted">
        Select an exoplanet to analyze its atmosphere for biosignatures.
      </p>

      <div className="mt-12">
        <PlanetTimeline
          planets={planets}
          selectedId={selectedId}
          onSelect={handleSelect}
        />
      </div>

      {selectedPlanet && (
        <div key={selectedPlanet.id} className="mt-10">
          <PlanetSummary planet={selectedPlanet} />
        </div>
      )}

      <AnalyzeButton
        disabled={!selectedId}
        loading={loading}
        onClick={handleAnalyze}
      />

      {results && (
        <div key={`results-${results.planetId}`} className="animate-fade-in">
          <ResultsPanel results={results} />
          <div className="pb-16">
            <BridgePanel />
          </div>
        </div>
      )}
    </div>
  );
}
