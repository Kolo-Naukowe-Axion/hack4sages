"use client";

import { useRef } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import type { Planet } from "@/types";

interface Props {
  planets: Planet[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export default function PlanetTimeline({ planets, selectedId, onSelect }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  const scroll = (dir: "left" | "right") => {
    scrollRef.current?.scrollBy({
      left: dir === "left" ? -320 : 320,
      behavior: "smooth",
    });
  };

  return (
    <div className="relative">
      <button
        onClick={() => scroll("left")}
        className="absolute -left-4 top-1/2 z-10 -translate-y-1/2 flex h-10 w-10 items-center justify-center rounded-full bg-deep text-muted shadow-lg transition-all hover:text-heading hover:shadow-xl"
        aria-label="Scroll left"
      >
        <ChevronLeft size={18} />
      </button>
      <button
        onClick={() => scroll("right")}
        className="absolute -right-4 top-1/2 z-10 -translate-y-1/2 flex h-10 w-10 items-center justify-center rounded-full bg-deep text-muted shadow-lg transition-all hover:text-heading hover:shadow-xl"
        aria-label="Scroll right"
      >
        <ChevronRight size={18} />
      </button>

      <div
        ref={scrollRef}
        className="no-scrollbar snap-scroll overflow-x-auto px-2 py-2"
      >
        <div className="flex gap-3" style={{ minWidth: "max-content" }}>
          {planets.map((planet) => {
            const isSelected = planet.id === selectedId;
            return (
              <button
                key={planet.id}
                onClick={() => onSelect(planet.id)}
                aria-pressed={isSelected}
                className={`group relative flex w-44 shrink-0 flex-col rounded-2xl border p-4 text-left transition-all duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan/30 ${
                  isSelected
                    ? "border-cyan bg-cyan/5 shadow-lg shadow-cyan/10"
                    : "border-border bg-deep hover:border-muted hover:shadow-md"
                }`}
              >
                <span
                  className={`font-display text-base font-semibold leading-tight transition-colors ${
                    isSelected ? "text-heading" : "text-heading group-hover:text-heading"
                  }`}
                >
                  {planet.name}
                </span>
                <span className="mt-1 text-xs text-muted">
                  {planet.starSystem}
                </span>

                <div className="mt-3 flex items-center justify-between">
                  <span className="font-mono text-xs text-muted">
                    {planet.eqTempK ? `${planet.eqTempK} K` : `${planet.radiusEarth.toFixed(1)} R⊕`}
                  </span>
                  {planet.inHabitableZone && (
                    <span className="flex h-5 items-center rounded-full bg-green/10 px-2 text-[10px] font-medium text-green">
                      HZ
                    </span>
                  )}
                </div>

                <span className="mt-2 font-mono text-[10px] text-muted/60">
                  {planet.discoveryYear}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
