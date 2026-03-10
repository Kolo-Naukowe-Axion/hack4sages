"use client";

import { useState, useEffect } from "react";
import { Satellite } from "lucide-react";
import { ResponsiveLine } from "@nivo/line";
import type { Planet } from "@/types";

interface Props {
  planet: Planet;
}

const statRow = (label: string, value: string | null, unit: string) => (
  <div className="flex items-baseline justify-between py-3 border-b border-border/40 last:border-0">
    <span className="text-sm text-muted">{label}</span>
    <span className="font-mono text-sm font-medium text-heading">
      {value ?? "—"}{value ? ` ${unit}` : ""}
    </span>
  </div>
);

export default function PlanetSummary({ planet }: Props) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const chartData = [
    {
      id: "spectrum",
      data: planet.spectrumData.map((d) => ({
        x: d.wavelength,
        y: d.flux,
      })),
    },
  ];

  return (
    <div className="animate-fade-in rounded-2xl bg-deep p-8 shadow-sm lg:p-10">
      <div className="grid gap-10 lg:grid-cols-[55fr_45fr]">
        <div>
          <h2 className="font-display text-3xl font-semibold tracking-tight text-heading">
            {planet.name}
          </h2>
          <p className="mt-1 text-sm text-muted">{planet.starSystem} system</p>

          <div className="mt-8">
            {statRow("Mass", planet.massEarth?.toFixed(2) ?? null, "M⊕")}
            {statRow("Radius", planet.radiusEarth.toFixed(2), "R⊕")}
            {statRow("Temperature", planet.eqTempK?.toString() ?? null, "K")}
            {statRow("Orbital Period", planet.orbitalPeriodDays.toFixed(1), "days")}
            {statRow("Distance", planet.distanceLy.toString(), "ly")}
          </div>

          {planet.inHabitableZone && (
            <div className="mt-5">
              <span className="inline-flex items-center rounded-full bg-green/10 px-3.5 py-1.5 text-xs font-medium text-green">
                Habitable Zone
              </span>
            </div>
          )}
        </div>

        <div>
          <div className="h-[280px]">
            {!mounted ? (
              <div className="flex h-full items-center justify-center">
                <span className="text-sm text-muted">Loading chart...</span>
              </div>
            ) : (
            <ResponsiveLine
              data={chartData}
              margin={{ top: 16, right: 20, bottom: 44, left: 54 }}
              xScale={{ type: "linear", min: "auto", max: "auto" }}
              yScale={{ type: "linear", min: "auto", max: "auto" }}
              curve="monotoneX"
              enableArea={true}
              areaOpacity={0.06}
              colors={["#0071e3"]}
              lineWidth={2}
              enablePoints={false}
              enableGridX={false}
              gridYValues={4}
              axisBottom={{
                tickSize: 0,
                tickPadding: 10,
                tickValues: 5,
                format: (v: number) => `${v}μm`,
              }}
              axisLeft={{
                tickSize: 0,
                tickPadding: 10,
                tickValues: 4,
                format: (v: number) => v.toFixed(3),
              }}
              theme={{
                axis: {
                  ticks: {
                    text: {
                      fill: "#86868b",
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                    },
                  },
                },
                grid: {
                  line: {
                    stroke: "#e5e5e5",
                    strokeWidth: 1,
                  },
                },
                crosshair: {
                  line: {
                    stroke: "#0071e3",
                    strokeWidth: 1,
                    strokeOpacity: 0.4,
                  },
                },
              }}
              tooltip={({ point }) => (
                <div className="rounded-xl border border-border bg-deep px-3.5 py-2 text-xs shadow-lg">
                  <span className="text-muted">λ = </span>
                  <span className="font-mono font-medium text-heading">
                    {Number(point.data.x).toFixed(3)} μm
                  </span>
                  <span className="mx-2 text-border">|</span>
                  <span className="text-muted">flux = </span>
                  <span className="font-mono font-medium text-heading">
                    {Number(point.data.y).toFixed(4)}
                  </span>
                </div>
              )}
              useMesh={true}
            />
            )}
          </div>

          <div className="mt-4 flex items-center gap-2">
            {planet.hasJWSTData ? (
              <>
                <Satellite size={14} className="text-green" />
                <span className="text-xs font-medium text-green">JWST Data</span>
              </>
            ) : (
              <>
                <Satellite size={14} className="text-muted" />
                <span className="text-xs text-muted">Synthetic Spectrum</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
