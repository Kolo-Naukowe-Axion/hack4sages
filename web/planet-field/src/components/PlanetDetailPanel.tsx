"use client";

import { Planet } from "@/types";
import { SpectrumChart } from "./SpectrumChart";

interface Props {
  planet: Planet;
  onAnalyze: () => void;
  analyzing: boolean;
}

const starTypeColors: Record<string, string> = {
  "M-dwarf": "text-accent-red bg-accent-red/10",
  "K-dwarf": "text-accent-amber bg-accent-amber/10",
  "G-type": "text-accent-amber bg-accent-amber/10",
  "F-type": "text-accent-blue bg-accent-blue/10",
};

function HabitabilityGauge({ score }: { score: number }) {
  const color =
    score >= 70 ? "#22c55e" : score >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <div className="flex items-center gap-2">
      <div className="w-full h-2 bg-border rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${score}%`, background: color }}
        />
      </div>
      <span className="text-xs font-mono text-text shrink-0">{score}</span>
    </div>
  );
}

export function PlanetDetailPanel({ planet, onAnalyze, analyzing }: Props) {
  const stats = [
    { label: "Eq. Temp", value: planet.eqTempK ? `${planet.eqTempK} K` : "N/A" },
    { label: "Radius", value: `${planet.radiusEarth} R\u2295` },
    { label: "Mass", value: planet.massEarth ? `${planet.massEarth} M\u2295` : "N/A" },
    { label: "Period", value: `${planet.orbitalPeriodDays} d` },
    { label: "Star System", value: planet.starSystem },
    { label: "Distance", value: `${planet.distanceLy} ly` },
    { label: "Data Source", value: planet.spectrumType === "jwst" ? "JWST" : "Synthetic" },
    { label: "Discovered", value: `${planet.discoveryYear}` },
  ];

  return (
    <div className="space-y-5">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold text-text">{planet.name}</h2>
        <span className={`text-[10px] uppercase tracking-wide font-medium px-2 py-0.5 rounded-md ${starTypeColors[planet.starType] || "text-muted bg-surface"}`}>
          {planet.starType}
        </span>
        {planet.inHabitableZone && (
          <span className="text-xs font-medium text-accent-green bg-accent-green/10 px-2 py-0.5 rounded-md">
            Habitable Zone
          </span>
        )}
        {planet.hasJWSTData && (
          <span className="text-xs font-medium text-accent-amber bg-accent-amber/10 px-2 py-0.5 rounded-md">
            JWST Data
          </span>
        )}
      </div>

      {/* Discovery context */}
      <p className="text-sm text-muted leading-relaxed">{planet.discoveryContext}</p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: stats + habitability */}
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {stats.map(({ label, value }) => (
              <div key={label} className="bg-surface border border-border rounded-lg px-3 py-2">
                <div className="text-[11px] text-muted mb-0.5">{label}</div>
                <div className="text-sm text-text font-mono font-medium">{value}</div>
              </div>
            ))}
          </div>

          {/* Habitability score */}
          <div className="bg-surface border border-border rounded-lg px-4 py-3">
            <div className="text-[11px] text-muted mb-2">Habitability Score</div>
            <HabitabilityGauge score={planet.habitabilityScore} />
          </div>
        </div>

        {/* Right: spectrum */}
        <div>
          <h3 className="text-[11px] uppercase tracking-wider text-muted mb-2">
            Transmission Spectrum
          </h3>
          <SpectrumChart data={planet.spectrumData} />
        </div>
      </div>

      {/* Analyze button */}
      <button
        onClick={onAnalyze}
        disabled={analyzing}
        className="px-6 py-2.5 rounded-md text-sm font-medium transition-colors
          bg-accent-blue text-white hover:bg-accent-blue/80
          disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {analyzing ? (
          <span className="flex items-center gap-2">
            <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Analyzing...
          </span>
        ) : (
          "Analyze Atmosphere"
        )}
      </button>
    </div>
  );
}
