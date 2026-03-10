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
      left: dir === "left" ? -300 : 300,
      behavior: "smooth",
    });
  };

  return (
    <div className="relative">
      <button
        onClick={() => scroll("left")}
        className="absolute left-3 top-1/2 z-10 -translate-y-1/2 flex h-10 w-10 items-center justify-center rounded-full border border-border bg-deep text-muted shadow-md transition-colors hover:text-heading"
        aria-label="Scroll left"
      >
        <ChevronLeft size={18} />
      </button>
      <button
        onClick={() => scroll("right")}
        className="absolute right-3 top-1/2 z-10 -translate-y-1/2 flex h-10 w-10 items-center justify-center rounded-full border border-border bg-deep text-muted shadow-md transition-colors hover:text-heading"
        aria-label="Scroll right"
      >
        <ChevronRight size={18} />
      </button>

      <div
        ref={scrollRef}
        className="no-scrollbar overflow-x-auto rounded-2xl bg-surface px-14 py-10"
        style={{ scrollbarWidth: "none" }}
      >
        <div className="relative flex items-center gap-8" style={{ minWidth: "max-content" }}>
          <div className="pointer-events-none absolute left-8 right-8 top-1/2 h-px -translate-y-1/2 bg-border" />

          {planets.map((planet) => {
            const isSelected = planet.id === selectedId;
            return (
              <button
                key={planet.id}
                onClick={() => onSelect(planet.id)}
                className="group relative z-10 flex flex-col items-center gap-2 focus:outline-none"
              >
                <span className="font-mono text-xs text-muted transition-colors group-hover:text-heading">
                  {planet.discoveryYear}
                </span>

                <div
                  className={`relative flex h-16 w-16 items-center justify-center rounded-full border-2 font-display text-lg font-semibold transition-all duration-300 ${
                    isSelected
                      ? "border-cyan bg-cyan/10 text-cyan shadow-lg shadow-cyan/15 scale-110"
                      : "border-border bg-deep text-muted group-hover:border-cyan group-hover:text-heading group-hover:shadow-md"
                  }`}
                >
                  {planet.name.charAt(0)}
                  {planet.inHabitableZone && (
                    <span className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-surface bg-green" />
                  )}
                </div>

                <span
                  className={`max-w-[5rem] truncate text-center text-sm transition-colors ${
                    isSelected ? "text-cyan" : "text-text group-hover:text-heading"
                  }`}
                >
                  {planet.name}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
