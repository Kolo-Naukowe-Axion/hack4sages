"use client";

import { ResponsiveLine } from "@nivo/line";

interface Props {
  data: { wavelength: number; flux: number }[];
}

export function SpectrumChart({ data }: Props) {
  const chartData = [
    {
      id: "Spectrum",
      data: data.map((d) => ({ x: d.wavelength, y: d.flux })),
    },
  ];

  return (
    <div className="h-[320px] w-full">
      <ResponsiveLine
        data={chartData}
        margin={{ top: 20, right: 20, bottom: 50, left: 60 }}
        xScale={{ type: "linear", min: "auto", max: "auto" }}
        yScale={{ type: "linear", min: "auto", max: "auto" }}
        curve="monotoneX"
        enableArea={true}
        areaOpacity={0.08}
        colors={["#a855f7"]}
        lineWidth={2}
        enablePoints={false}
        enableGridX={false}
        gridYValues={5}
        theme={{
          background: "transparent",
          text: { fill: "#a1a1aa", fontSize: 11, fontFamily: "JetBrains Mono, monospace" },
          grid: { line: { stroke: "#27272a", strokeWidth: 1 } },
          axis: {
            ticks: { line: { stroke: "#27272a" }, text: { fill: "#a1a1aa" } },
            legend: { text: { fill: "#a1a1aa", fontSize: 12 } },
          },
          crosshair: { line: { stroke: "#a855f7", strokeWidth: 1, strokeOpacity: 0.5 } },
          tooltip: {
            container: {
              background: "#18181b",
              color: "#fafafa",
              fontSize: 12,
              borderRadius: "8px",
              border: "1px solid #27272a",
              boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
            },
          },
        }}
        axisBottom={{
          tickSize: 0,
          tickPadding: 12,
          legend: "Wavelength (μm)",
          legendOffset: 40,
          legendPosition: "middle",
        }}
        axisLeft={{
          tickSize: 0,
          tickPadding: 12,
          legend: "Transit Depth",
          legendOffset: -48,
          legendPosition: "middle",
        }}
        useMesh={true}
      />
    </div>
  );
}
