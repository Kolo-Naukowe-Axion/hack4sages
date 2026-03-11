"use client";

import { useState, useCallback } from "react";
import { planets } from "@/data/planets";
import { getMockResults } from "@/data/mockResults";
import { PlanetResults } from "@/types";
import { useScrollReveal } from "@/hooks/useScrollReveal";
import { PlanetCard } from "@/components/PlanetCard";
import { SpectrumChart } from "@/components/SpectrumChart";
import { ResultCard } from "@/components/ResultCard";
import { Footer } from "@/components/Footer";
import {
  Loader2,
  Star,
  Thermometer,
  Ruler,
  Globe,
  Clock,
  Calendar,
  Orbit,
  Database,
} from "lucide-react";

export default function ExplorerPage() {
  const ref = useScrollReveal();
  const [selectedId, setSelectedId] = useState(planets[0].id);
  const [results, setResults] = useState<PlanetResults | null>(null);
  const [loading, setLoading] = useState(false);

  const planet = planets.find((p) => p.id === selectedId)!;

  const analyze = useCallback(async () => {
    setLoading(true);
    setResults(null);
    const r = await getMockResults(selectedId);
    setResults(r);
    setLoading(false);
  }, [selectedId]);

  const details = [
    { icon: <Star className="w-3.5 h-3.5" />, label: "Star System", value: planet.starSystem },
    { icon: <Globe className="w-3.5 h-3.5" />, label: "Star Type", value: planet.starType },
    {
      icon: <Thermometer className="w-3.5 h-3.5" />,
      label: "Eq. Temp",
      value: planet.eqTempK ? `${planet.eqTempK} K` : "Unknown",
    },
    {
      icon: <Ruler className="w-3.5 h-3.5" />,
      label: "Radius",
      value: `${planet.radiusEarth} R⊕`,
    },
    {
      icon: <Orbit className="w-3.5 h-3.5" />,
      label: "Mass",
      value: planet.massEarth ? `${planet.massEarth} M⊕` : "Unknown",
    },
    {
      icon: <Clock className="w-3.5 h-3.5" />,
      label: "Orbital Period",
      value: `${planet.orbitalPeriodDays} days`,
    },
    {
      icon: <Calendar className="w-3.5 h-3.5" />,
      label: "Discovered",
      value: `${planet.discoveryYear}`,
    },
    {
      icon: <Database className="w-3.5 h-3.5" />,
      label: "Data Source",
      value: planet.spectrumType === "jwst" ? "JWST" : "Synthetic",
    },
  ];

  return (
    <div ref={ref}>
      <div className="pt-24 pb-16 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="mb-10">
            <h1 className="reveal text-3xl sm:text-4xl font-bold mb-2">
              Planet <span className="gradient-text">Explorer</span>
            </h1>
            <p className="reveal reveal-delay-1 text-muted text-sm">
              Select an exoplanet to view its spectrum and run biosignature analysis.
            </p>
          </div>

          <div className="reveal reveal-delay-2 grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-6">
            <div className="space-y-2 max-h-[calc(100vh-200px)] overflow-y-auto pr-1">
              {planets.map((p) => (
                <PlanetCard
                  key={p.id}
                  planet={p}
                  selected={p.id === selectedId}
                  onClick={() => {
                    setSelectedId(p.id);
                    setResults(null);
                  }}
                />
              ))}
            </div>

            <div className="space-y-6">
              <div className="p-6 rounded-xl border border-border bg-surface/50">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className="text-xl font-bold">{planet.name}</h2>
                    <p className="text-xs text-muted mt-1">{planet.discoveryContext}</p>
                  </div>
                  <div className="flex gap-2">
                    {planet.inHabitableZone && (
                      <span className="text-xs px-2.5 py-1 rounded-lg bg-green/10 text-green font-medium">
                        Habitable Zone
                      </span>
                    )}
                    {planet.hasJWSTData && (
                      <span className="text-xs px-2.5 py-1 rounded-lg bg-purple/10 text-purple font-medium">
                        JWST Data
                      </span>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                  {details.map((d) => (
                    <div
                      key={d.label}
                      className="flex items-center gap-2 text-xs text-muted"
                    >
                      <span className="text-muted/60">{d.icon}</span>
                      <div>
                        <div className="text-[10px] text-muted/60">{d.label}</div>
                        <div className="text-text font-mono text-xs">{d.value}</div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="border border-border rounded-xl p-4 bg-bg">
                  <SpectrumChart data={planet.spectrumData} />
                </div>
              </div>

              <div className="flex justify-center">
                <button
                  onClick={analyze}
                  disabled={loading}
                  className="gradient-bg inline-flex items-center gap-2 px-8 py-3 text-sm font-semibold rounded-xl text-white hover:opacity-90 transition-opacity disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Analyzing atmosphere...
                    </>
                  ) : (
                    "Analyze Atmosphere"
                  )}
                </button>
              </div>

              {results && (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {results.results.map((r) => (
                    <ResultCard key={r.modelName} result={r} />
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}
