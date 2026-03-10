"use client";

import { ResponsiveLine } from "@nivo/line";
import type { Planet } from "@/types";

interface Props {
  planet: Planet;
}

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

  const rows: [string, string][] = [
    ["Mass", planet.massEarth ? `${planet.massEarth.toFixed(2)} M⊕` : "—"],
    ["Radius", `${planet.radiusEarth.toFixed(2)} R⊕`],
    ["Eq. Temperature", planet.eqTempK ? `${planet.eqTempK} K` : "—"],
    ["Orbital Period", `${planet.orbitalPeriodDays.toFixed(1)} d`],
    ["Distance", `${planet.distanceLy} ly`],
    ["Habitable Zone", planet.inHabitableZone ? "Yes" : "No"],
    ["Spectrum Source", planet.hasJWSTData ? "JWST" : "Synthetic"],
  ];

  return (
    <div className="animate-fade-in">
      <div className="grid gap-8 lg:grid-cols-2">
        {/* Left: Data Table */}
        <div>
          <p className="figure-caption mb-4">
            <strong>Table 1.</strong> Physical and orbital parameters of {planet.name} ({planet.starSystem} system).
          </p>
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b-2 border-heading">
                <th className="py-2 pr-4 text-left font-serif font-semibold text-heading">Parameter</th>
                <th className="py-2 text-left font-serif font-semibold text-heading">Value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([label, value]) => (
                <tr key={label} className="border-b border-border-light">
                  <td className="py-2 pr-4 font-sans text-muted">{label}</td>
                  <td className="py-2 font-mono text-sm text-text">{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Right: Spectrum Chart */}
        <div>
          <div className="border border-border bg-paper p-4">
            <div className="h-[260px]">
              <ResponsiveLine
                data={chartData}
                margin={{ top: 12, right: 16, bottom: 44, left: 56 }}
                xScale={{ type: "linear", min: "auto", max: "auto" }}
                yScale={{ type: "linear", min: "auto", max: "auto" }}
                curve="monotoneX"
                enableArea={true}
                areaOpacity={0.06}
                colors={["#1e3a5f"]}
                lineWidth={1.5}
                enablePoints={false}
                enableGridX={true}
                enableGridY={true}
                gridYValues={4}
                gridXValues={5}
                axisBottom={{
                  tickSize: 4,
                  tickPadding: 6,
                  tickValues: 5,
                  legend: "Wavelength (μm)",
                  legendOffset: 36,
                  legendPosition: "middle" as const,
                  format: (v: number) => v.toFixed(1),
                }}
                axisLeft={{
                  tickSize: 4,
                  tickPadding: 6,
                  tickValues: 4,
                  legend: "Normalized Flux",
                  legendOffset: -46,
                  legendPosition: "middle" as const,
                  format: (v: number) => v.toFixed(3),
                }}
                theme={{
                  axis: {
                    ticks: {
                      text: {
                        fill: "#6b6458",
                        fontFamily: "var(--font-mono)",
                        fontSize: 10,
                      },
                    },
                    legend: {
                      text: {
                        fill: "#3d3531",
                        fontFamily: "var(--font-sans)",
                        fontSize: 11,
                        fontWeight: 500,
                      },
                    },
                  },
                  grid: {
                    line: {
                      stroke: "#e0dbd4",
                      strokeWidth: 1,
                      strokeDasharray: "3 3",
                    },
                  },
                  crosshair: {
                    line: {
                      stroke: "#1e3a5f",
                      strokeWidth: 1,
                      strokeOpacity: 0.4,
                    },
                  },
                }}
                tooltip={({ point }) => (
                  <div className="border border-border bg-paper px-3 py-1.5 text-xs shadow-sm">
                    <span className="text-muted">λ = </span>
                    <span className="font-mono text-accent">{Number(point.data.x).toFixed(3)} μm</span>
                    <span className="mx-1.5 text-border">|</span>
                    <span className="text-muted">F = </span>
                    <span className="font-mono text-heading">{Number(point.data.y).toFixed(4)}</span>
                  </div>
                )}
                useMesh={true}
              />
            </div>
          </div>
          <p className="figure-caption mt-3">
            <strong>Figure 1.</strong> Transmission spectrum of {planet.name}.{" "}
            {planet.hasJWSTData
              ? "Data obtained from JWST observations."
              : "Synthetic spectrum generated from atmospheric models."}
          </p>
        </div>
      </div>
    </div>
  );
}
