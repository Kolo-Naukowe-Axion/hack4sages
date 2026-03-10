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
        colors={["#00d4ff"]}
        lineWidth={2}
        pointSize={0}
        enableArea
        areaOpacity={0.08}
        enableGridX={false}
        gridYValues={4}
        axisBottom={{
          tickValues: [1, 2, 3, 4, 5],
          legend: "Wavelength (um)",
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
          text: { fill: "rgba(255,255,255,0.5)", fontSize: 10 },
          axis: {
            ticks: { line: { stroke: "rgba(255,255,255,0.1)" }, text: { fill: "rgba(255,255,255,0.4)" } },
            legend: { text: { fill: "rgba(255,255,255,0.4)", fontSize: 10 } },
          },
          grid: { line: { stroke: "rgba(255,255,255,0.05)" } },
          crosshair: { line: { stroke: "#00d4ff", strokeWidth: 1 } },
        }}
        enableCrosshair
        useMesh
        markers={highlights.map((h) => ({
          axis: "x" as const,
          value: (h.start + h.end) / 2,
          lineStyle: { stroke: "#ffaa00", strokeWidth: 1, strokeDasharray: "4 3" },
          legend: h.gas,
          legendPosition: "top" as const,
          textStyle: { fill: "#ffaa00", fontSize: 9 },
        }))}
      />
    </div>
  );
}
