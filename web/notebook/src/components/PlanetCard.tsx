import { Planet } from "@/types";
import { Globe, Telescope } from "lucide-react";

interface PlanetCardProps {
  planet: Planet;
  selected: boolean;
  onClick: () => void;
}

export default function PlanetCard({ planet, selected, onClick }: PlanetCardProps) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-3 rounded border transition-all ${
        selected
          ? "border-nb-blue bg-[#e3f2fd] shadow-sm"
          : "border-nb-border bg-white hover:border-nb-blue/40 hover:bg-nb-hover"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[13px] font-semibold text-nb-text truncate">
              {planet.name}
            </span>
            {planet.hasJWSTData && (
              <Telescope size={12} className="text-nb-blue shrink-0" />
            )}
            {planet.inHabitableZone && (
              <Globe size={12} className="text-nb-green shrink-0" />
            )}
          </div>
          <div className="text-[11px] text-nb-muted mt-0.5 font-mono">
            {planet.starSystem} &middot; {planet.distanceLy} ly
          </div>
        </div>
        <div className="text-right shrink-0">
          <div className="text-[11px] font-mono text-nb-muted">
            {planet.radiusEarth.toFixed(2)} R<sub>E</sub>
          </div>
          <div className="text-[11px] font-mono text-nb-muted">
            {planet.eqTempK ?? "?"} K
          </div>
        </div>
      </div>
    </button>
  );
}
