"use client";

import { ResponsiveLine } from "@nivo/line";
import { ModelResult } from "@/types";

interface SpectrumChartProps {
  data: { wavelength: number; flux: number }[];
  highlights?: ModelResult["spectrumHighlights"];
}

export default function SpectrumChart({ data, highlights = [] }: SpectrumChartProps) {
  const chartData = [
    {
      id: "Transmission Spectrum",
      data: data.map((d) => ({ x: d.wavelength, y: d.flux })),
    },
  ];

  return (
    <div className="h-[280px] w-full bg-white border border-nb-border rounded p-2">
      <ResponsiveLine
        data={chartData}
        margin={{ top: 20, right: 20, bottom: 50, left: 60 }}
        xScale={{ type: "linear", min: "auto", max: "auto" }}
        yScale={{ type: "linear", min: "auto", max: "auto" }}
        curve="monotoneX"
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          legend: "Wavelength (um)",
          legendOffset: 40,
          legendPosition: "middle",
        }}
        axisLeft={{
          tickSize: 5,
          tickPadding: 5,
          legend: "Relative Flux",
          legendOffset: -50,
          legendPosition: "middle",
        }}
        colors={["#2196f3"]}
        lineWidth={2}
        pointSize={4}
        pointColor="#2196f3"
        pointBorderWidth={1}
        pointBorderColor="#ffffff"
        enableArea={true}
        areaOpacity={0.08}
        useMesh={true}
        theme={{
          axis: {
            ticks: {
              text: { fontSize: 11, fontFamily: "var(--font-mono)", fill: "#757575" },
            },
            legend: {
              text: { fontSize: 12, fontFamily: "var(--font-sans)", fill: "#212121" },
            },
          },
          grid: {
            line: { stroke: "#e0e0e0", strokeWidth: 1 },
          },
          crosshair: {
            line: { stroke: "#2196f3", strokeWidth: 1, strokeDasharray: "4 4" },
          },
        }}
        markers={highlights.map((h) => ({
          axis: "x" as const,
          value: (h.start + h.end) / 2,
          lineStyle: { stroke: "#ff9800", strokeWidth: 2, strokeDasharray: "4 4" },
          legend: h.gas,
          textStyle: { fontSize: 10, fill: "#ff9800", fontFamily: "var(--font-mono)" },
        }))}
      />
    </div>
  );
}
