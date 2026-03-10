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
        className="absolute left-2 top-1/2 z-10 -translate-y-1/2 flex h-8 w-8 items-center justify-center rounded-full border border-border bg-paper text-muted shadow-sm transition-colors hover:text-heading"
        aria-label="Scroll left"
      >
        <ChevronLeft size={16} />
      </button>
      <button
        onClick={() => scroll("right")}
        className="absolute right-2 top-1/2 z-10 -translate-y-1/2 flex h-8 w-8 items-center justify-center rounded-full border border-border bg-paper text-muted shadow-sm transition-colors hover:text-heading"
        aria-label="Scroll right"
      >
        <ChevronRight size={16} />
      </button>

      <div
        ref={scrollRef}
        className="no-scrollbar overflow-x-auto border border-border bg-paper px-12 py-6"
        style={{ scrollbarWidth: "none" }}
      >
        <div className="relative flex items-center gap-6" style={{ minWidth: "max-content" }}>
          <div className="pointer-events-none absolute left-6 right-6 top-1/2 h-px -translate-y-1/2 bg-border-light" />

          {planets.map((planet) => {
            const isSelected = planet.id === selectedId;
            return (
              <button
                key={planet.id}
                onClick={() => onSelect(planet.id)}
                aria-label={`${planet.name}, discovered ${planet.discoveryYear}${planet.inHabitableZone ? ", in habitable zone" : ""}`}
                aria-pressed={isSelected}
                className="group relative z-10 flex flex-col items-center gap-1.5 focus:outline-none"
              >
                <span className="font-mono text-[10px] text-muted transition-colors group-hover:text-heading">
                  {planet.discoveryYear}
                </span>

                <div
                  className={`relative flex h-11 w-11 items-center justify-center rounded-full border font-serif text-sm font-semibold transition-all duration-200 ${
                    isSelected
                      ? "border-accent bg-accent/8 text-accent shadow-sm"
                      : "border-border bg-paper text-muted group-hover:border-accent/50 group-hover:text-heading"
                  }`}
                >
                  {planet.name.charAt(0)}
                  {planet.inHabitableZone && (
                    <span aria-hidden="true" className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-paper bg-green" />
                  )}
                </div>

                <span
                  className={`max-w-[4.5rem] truncate text-center font-sans text-xs transition-colors ${
                    isSelected ? "font-medium text-accent" : "text-muted group-hover:text-heading"
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
