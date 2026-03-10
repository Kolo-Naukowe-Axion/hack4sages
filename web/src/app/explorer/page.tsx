"use client";

import { useState } from "react";
import { planets } from "@/data/planets";
import PlanetTimeline from "@/components/PlanetTimeline";
import PlanetSummary from "@/components/PlanetSummary";

export default function ExplorerPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selectedPlanet = planets.find((p) => p.id === selectedId) ?? null;

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
          onSelect={setSelectedId}
        />
      </div>

      {selectedPlanet && (
        <div key={selectedPlanet.id} className="mt-8">
          <PlanetSummary planet={selectedPlanet} />
        </div>
      )}
    </div>
  );
}
