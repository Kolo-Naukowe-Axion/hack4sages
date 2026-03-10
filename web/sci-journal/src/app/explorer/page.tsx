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
    <article className="mx-auto max-w-4xl px-6 py-12 lg:py-16">
      <header>
        <p className="section-number text-sm">§3</p>
        <h1 className="font-serif text-3xl font-bold text-heading lg:text-4xl">
          Planet Explorer
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-muted">
          Select an exoplanet from the catalog below, review its physical parameters
          and transmission spectrum, then run the biosignature detection pipeline.
        </p>
      </header>

      <hr className="journal-rule mt-6" />

      <div className="mt-8">
        <p className="figure-caption mb-3">
          Exoplanet catalog. Select a target to view its parameters and spectrum. Green indicators denote planets in the habitable zone.
        </p>
        <PlanetTimeline
          planets={planets}
          selectedId={selectedId}
          onSelect={handleSelect}
        />
      </div>

      {selectedPlanet && (
        <div key={selectedPlanet.id} className="mt-10">
          <hr className="journal-rule mb-8" />
          <p className="section-number text-sm">§3.1</p>
          <h2 className="font-serif text-2xl font-semibold text-heading">
            {selectedPlanet.name}
          </h2>
          <p className="mt-1 text-sm text-muted">
            {selectedPlanet.starSystem} system &middot; Discovered {selectedPlanet.discoveryYear}
          </p>

          <div className="mt-6">
            <PlanetSummary planet={selectedPlanet} />
          </div>
        </div>
      )}

      <AnalyzeButton
        disabled={!selectedId}
        loading={loading}
        onClick={handleAnalyze}
      />

      {results && (
        <div key={`results-${results.planetId}`} className="animate-fade-in">
          <hr className="journal-rule" />
          <ResultsPanel results={results} />
          <BridgePanel />
        </div>
      )}
    </article>
  );
}
