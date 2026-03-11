import { Planet } from "@/types";
import { Star, Thermometer, Ruler, Globe } from "lucide-react";

interface Props {
  planet: Planet;
  selected: boolean;
  onClick: () => void;
}

export function PlanetCard({ planet, selected, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-4 rounded-xl border transition-all duration-200 ${
        selected
          ? "border-purple bg-purple/5"
          : "border-border bg-surface/50 hover:border-muted"
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold">{planet.name}</span>
        <div className="flex items-center gap-1.5">
          {planet.hasJWSTData && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple/20 text-purple font-medium">
              JWST
            </span>
          )}
          {planet.inHabitableZone && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-green/20 text-green font-medium">
              HZ
            </span>
          )}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-y-1.5 text-[11px] text-muted">
        <div className="flex items-center gap-1.5">
          <Star className="w-3 h-3" />
          {planet.starSystem}
        </div>
        <div className="flex items-center gap-1.5">
          <Thermometer className="w-3 h-3" />
          {planet.eqTempK ? `${planet.eqTempK} K` : "—"}
        </div>
        <div className="flex items-center gap-1.5">
          <Ruler className="w-3 h-3" />
          {planet.radiusEarth} R⊕
        </div>
        <div className="flex items-center gap-1.5">
          <Globe className="w-3 h-3" />
          {planet.distanceLy} ly
        </div>
      </div>
    </button>
  );
}
