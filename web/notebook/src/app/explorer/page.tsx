"use client";

import { useState } from "react";
import Cell from "@/components/Cell";
import CodeBlock from "@/components/CodeBlock";
import PlanetCard from "@/components/PlanetCard";
import SpectrumChart from "@/components/SpectrumChart";
import ResultsOutput from "@/components/ResultsOutput";
import { useScrollReveal } from "@/hooks/useScrollReveal";
import { planets } from "@/data/planets";
import { getMockResults } from "@/data/mockResults";
import { Planet, PlanetResults } from "@/types";
import { Play, Globe, Telescope } from "lucide-react";

export default function ExplorerPage() {
  const ref = useScrollReveal();
  const [selected, setSelected] = useState<Planet | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<PlanetResults | null>(null);

  async function runAnalysis() {
    if (!selected) return;
    setAnalyzing(true);
    setResults(null);
    const r = await getMockResults(selected.id);
    setResults(r);
    setAnalyzing(false);
  }

  return (
    <div ref={ref} className="max-w-[900px] mx-auto px-4 py-6 space-y-6 cell-stagger">
      {/* Title — no cell */}
      <div className="reveal space-y-1">
        <h1 className="text-2xl font-bold text-nb-text">Planet Explorer</h1>
        <p className="text-nb-muted text-sm">
          Browse the exoplanet catalog, inspect transmission spectra, and run
          biosignature analysis.
        </p>
      </div>

      {/* Code cell: load catalog */}
      <div className="reveal">
        <Cell type="code" executionCount={1}>
          <CodeBlock
            lines={[
              "planets = exobiome.load_catalog()",
              `print(f"Loaded {len(planets)} exoplanets")`,
              "planets.summary()",
            ]}
          />
        </Cell>
      </div>

      {/* Planet grid — free flowing */}
      <div className="reveal">
        <div className="flex items-center justify-between mb-3">
          <div className="font-mono text-[12px] text-nb-muted">
            # {planets.length} exoplanets loaded
          </div>
          <div className="flex items-center gap-3 text-[11px] font-mono text-nb-muted">
            <span className="flex items-center gap-1">
              <Telescope size={11} className="text-nb-blue" /> JWST data
            </span>
            <span className="flex items-center gap-1">
              <Globe size={11} className="text-nb-green" /> Habitable zone
            </span>
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {planets.map((p) => (
            <PlanetCard
              key={p.id}
              planet={p}
              selected={selected?.id === p.id}
              onClick={() => {
                setSelected(p);
                setResults(null);
              }}
            />
          ))}
        </div>
      </div>

      {/* Selected planet detail */}
      {selected && (
        <>
          <Cell type="code" executionCount={2}>
            <CodeBlock
              lines={[
                `planet = planets["${selected.name}"]`,
                "planet.show_summary()",
              ]}
            />
          </Cell>

          {/* Planet details — free flowing */}
          <div className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <DetailItem label="Star System" value={selected.starSystem} />
              <DetailItem label="Discovery" value={String(selected.discoveryYear)} />
              <DetailItem label="Radius" value={`${selected.radiusEarth.toFixed(2)} R_Earth`} />
              <DetailItem label="Mass" value={selected.massEarth ? `${selected.massEarth.toFixed(2)} M_Earth` : "Unknown"} />
              <DetailItem label="Eq. Temp" value={selected.eqTempK ? `${selected.eqTempK} K` : "Unknown"} />
              <DetailItem label="Orbital Period" value={`${selected.orbitalPeriodDays} days`} />
              <DetailItem label="Distance" value={`${selected.distanceLy} ly`} />
              <DetailItem
                label="Spectrum"
                value={selected.spectrumType === "jwst" ? "JWST Observed" : "Synthetic (MultiREx)"}
              />
            </div>
            <div>
              <div className="font-mono text-[12px] text-nb-muted mb-2">
                # Transmission spectrum ({selected.spectrumData.length} data points)
              </div>
              <SpectrumChart data={selected.spectrumData} />
            </div>
          </div>

          {/* Run analysis cell */}
          <Cell type="code" executionCount={analyzing ? "*" : results ? 3 : null}>
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <CodeBlock
                  lines={[
                    `results = exobiome.analyze("${selected.name}")`,
                    "results.show()",
                  ]}
                />
              </div>
              <button
                onClick={runAnalysis}
                disabled={analyzing}
                className={`shrink-0 flex items-center gap-2 px-4 py-2 rounded font-mono text-[13px] transition-colors ${
                  analyzing
                    ? "bg-nb-running/20 text-nb-running cursor-wait"
                    : "bg-nb-green text-white hover:bg-green-600 cursor-pointer"
                }`}
              >
                <Play size={14} />
                {analyzing ? "Running..." : "Run Cell"}
              </button>
            </div>
          </Cell>

          {analyzing && (
            <div className="font-mono text-[13px] text-nb-running cell-running px-1">
              Encoding spectrum features... Initializing quantum reservoir...
            </div>
          )}

          {results && !analyzing && (
            <Cell type="output" executionCount={3}>
              <ResultsOutput results={results.results} />
            </Cell>
          )}
        </>
      )}
    </div>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-nb-card rounded-lg p-2.5 border border-nb-border">
      <div className="text-[10px] font-mono text-nb-muted uppercase tracking-wide">{label}</div>
      <div className="text-[13px] font-mono text-nb-text mt-0.5">{value}</div>
    </div>
  );
}
