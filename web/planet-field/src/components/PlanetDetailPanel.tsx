"use client";

import { Planet } from "@/types";
import { SpectrumChart } from "./SpectrumChart";
import {
  Thermometer,
  Ruler,
  Scale,
  Clock,
  Star,
  MapPin,
  Satellite,
  ShieldCheck,
} from "lucide-react";

interface Props {
  planet: Planet;
  onAnalyze: () => void;
  analyzing: boolean;
}

export function PlanetDetailPanel({ planet, onAnalyze, analyzing }: Props) {
  const stats = [
    { icon: Thermometer, label: "Eq. Temp", value: planet.eqTempK ? `${planet.eqTempK} K` : "N/A" },
    { icon: Ruler, label: "Radius", value: `${planet.radiusEarth} R\u2295` },
    { icon: Scale, label: "Mass", value: planet.massEarth ? `${planet.massEarth} M\u2295` : "N/A" },
    { icon: Clock, label: "Period", value: `${planet.orbitalPeriodDays} d` },
    { icon: Star, label: "Star System", value: planet.starSystem },
    { icon: MapPin, label: "Distance", value: `${planet.distanceLy} ly` },
  ];

  return (
    <div className="bg-space-800/60 backdrop-blur-md border border-white/8 rounded-2xl overflow-hidden">
      <div className="p-5 border-b border-white/5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xl font-bold text-white">{planet.name}</h2>
          <div className="flex items-center gap-2">
            {planet.inHabitableZone && (
              <span className="flex items-center gap-1 text-xs font-medium text-green bg-green-dim px-2.5 py-1 rounded-full">
                <ShieldCheck className="w-3 h-3" />
                Habitable Zone
              </span>
            )}
            {planet.hasJWSTData && (
              <span className="flex items-center gap-1 text-xs font-medium text-amber bg-amber-dim px-2.5 py-1 rounded-full">
                <Satellite className="w-3 h-3" />
                JWST Data
              </span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3">
          {stats.map(({ icon: Icon, label, value }) => (
            <div key={label} className="bg-white/3 rounded-lg px-3 py-2">
              <div className="flex items-center gap-1.5 text-white/40 text-[10px] uppercase tracking-wide mb-0.5">
                <Icon className="w-3 h-3" />
                {label}
              </div>
              <div className="text-sm text-white font-medium">{value}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="p-5 border-b border-white/5">
        <h3 className="text-xs uppercase tracking-wider text-white/40 mb-3">
          Transmission Spectrum
          <span className="ml-2 text-white/20">
            ({planet.spectrumType === "jwst" ? "JWST observed" : "Synthetic"})
          </span>
        </h3>
        <SpectrumChart data={planet.spectrumData} />
      </div>

      <div className="p-5">
        <button
          onClick={onAnalyze}
          disabled={analyzing}
          className="w-full py-3 rounded-xl font-semibold text-sm transition-all
            bg-gradient-to-r from-cyan to-green text-space-900
            hover:shadow-[0_0_24px_rgba(0,212,255,0.3)] hover:scale-[1.01]
            active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {analyzing ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-space-900/30 border-t-space-900 rounded-full animate-spin" />
              Analyzing Atmosphere...
            </span>
          ) : (
            "Analyze Atmosphere"
          )}
        </button>
      </div>
    </div>
  );
}
