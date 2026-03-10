"use client";

import { Satellite } from "lucide-react";
import { ResponsiveLine } from "@nivo/line";
import type { Planet } from "@/types";

interface Props {
  planet: Planet;
}

const statRow = (label: string, value: string | null, unit: string) => (
  <div className="flex items-baseline justify-between border-b border-border/50 py-2">
    <span className="text-sm text-muted">{label}</span>
    <span className="font-mono text-sm text-cyan">
      {value ?? "—"} {value ? unit : ""}
    </span>
  </div>
);

export default function PlanetSummary({ planet }: Props) {
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
    <div className="animate-fade-in rounded-2xl bg-deep p-8 shadow-lg shadow-black/5 lg:p-10">
      <div className="grid gap-10 lg:grid-cols-[55fr_45fr]">
        <div>
          <h2 className="font-display text-3xl font-semibold text-heading">
            {planet.name}
          </h2>
          <p className="mt-1 text-sm text-muted">{planet.starSystem} system</p>

          <div className="mt-6 space-y-0">
            {statRow("Mass", planet.massEarth?.toFixed(2) ?? null, "M⊕")}
            {statRow("Radius", planet.radiusEarth.toFixed(2), "R⊕")}
            {statRow("Temperature", planet.eqTempK?.toString() ?? null, "K")}
            {statRow("Orbital Period", planet.orbitalPeriodDays.toFixed(1), "days")}
            {statRow("Distance", planet.distanceLy.toString(), "ly")}
          </div>

          <div className="mt-4 flex items-center gap-3">
            {planet.inHabitableZone && (
              <span className="rounded-full bg-green/10 px-3 py-1 text-xs font-medium text-green">
                Habitable Zone
              </span>
            )}
          </div>
        </div>

        <div>
          <div className="h-[260px]">
            <ResponsiveLine
              data={chartData}
              margin={{ top: 10, right: 20, bottom: 40, left: 50 }}
              xScale={{ type: "linear", min: "auto", max: "auto" }}
              yScale={{ type: "linear", min: "auto", max: "auto" }}
              curve="monotoneX"
              enableArea={true}
              areaOpacity={0.08}
              colors={["#312e81"]}
              lineWidth={2}
              enablePoints={false}
              enableGridX={false}
              gridYValues={4}
              axisBottom={{
                tickSize: 0,
                tickPadding: 8,
                tickValues: 5,
                format: (v: number) => `${v}μm`,
              }}
              axisLeft={{
                tickSize: 0,
                tickPadding: 8,
                tickValues: 4,
                format: (v: number) => v.toFixed(3),
              }}
              theme={{
                axis: {
                  ticks: {
                    text: {
                      fill: "#78716c",
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                    },
                  },
                },
                grid: {
                  line: {
                    stroke: "#e7e5e4",
                    strokeWidth: 1,
                  },
                },
                crosshair: {
                  line: {
                    stroke: "#312e81",
                    strokeWidth: 1,
                    strokeOpacity: 0.5,
                  },
                },
              }}
              tooltip={({ point }) => (
                <div className="rounded-lg border border-border bg-deep px-3 py-2 text-xs shadow-lg">
                  <span className="text-muted">λ = </span>
                  <span className="font-mono text-cyan">{Number(point.data.x).toFixed(3)} μm</span>
                  <span className="mx-2 text-border">|</span>
                  <span className="text-muted">flux = </span>
                  <span className="font-mono text-heading">{Number(point.data.y).toFixed(4)}</span>
                </div>
              )}
              useMesh={true}
            />
          </div>

          <div className="mt-3 flex items-center gap-2">
            <Satellite size={14} className={planet.hasJWSTData ? "text-green" : "text-muted"} />
            {planet.hasJWSTData ? (
              <span className="text-xs font-medium text-green">Real JWST Data</span>
            ) : (
              <span className="rounded-full border border-border px-2 py-0.5 text-xs text-muted">
                Synthetic Spectrum
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
