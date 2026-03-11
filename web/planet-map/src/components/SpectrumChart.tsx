"use client";

import { ResponsiveLine } from "@nivo/line";

interface Props {
  data: { wavelength: number; flux: number }[];
  highlights?: { start: number; end: number; gas: string }[];
}

export function SpectrumChart({ data, highlights = [] }: Props) {
  const seriesData = data.map((d) => ({ x: d.wavelength, y: d.flux }));

  return (
    <div className="h-[220px] w-full">
      <ResponsiveLine
        data={[{ id: "spectrum", data: seriesData }]}
        margin={{ top: 12, right: 16, bottom: 40, left: 50 }}
        xScale={{ type: "linear", min: 0.6, max: 5.0 }}
        yScale={{ type: "linear", min: "auto", max: "auto" }}
        curve="monotoneX"
        colors={["#3b82f6"]}
        lineWidth={2}
        pointSize={0}
        enableArea
        areaOpacity={0.06}
        enableGridX={false}
        gridYValues={4}
        axisBottom={{
          tickValues: [1, 2, 3, 4, 5],
          legend: "Wavelength (\u00b5m)",
          legendOffset: 32,
          legendPosition: "middle",
        }}
        axisLeft={{
          tickValues: 4,
          legend: "Relative Flux",
          legendOffset: -40,
          legendPosition: "middle",
        }}
        theme={{
          text: { fill: "#9ca3af", fontSize: 10 },
          axis: {
            ticks: {
              line: { stroke: "#2a2d37" },
              text: { fill: "#9ca3af" },
            },
            legend: { text: { fill: "#9ca3af", fontSize: 10 } },
          },
          grid: { line: { stroke: "#2a2d37" } },
          crosshair: { line: { stroke: "#3b82f6", strokeWidth: 1 } },
        }}
        enableCrosshair
        useMesh
        markers={highlights.map((h) => ({
          axis: "x" as const,
          value: (h.start + h.end) / 2,
          lineStyle: {
            stroke: "#f59e0b",
            strokeWidth: 1,
            strokeDasharray: "4 3",
          },
          legend: h.gas,
          legendPosition: "top" as const,
          textStyle: { fill: "#f59e0b", fontSize: 9 },
        }))}
      />
    </div>
  );
}
