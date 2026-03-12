import { spectrumSeries } from "@/app/lib/project-data";

export function MiniSpectrum() {
  const width = 520;
  const height = 220;
  const padding = 22;
  const minX = spectrumSeries[0]?.wavelength ?? 0;
  const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 1;
  const values = spectrumSeries.flatMap((item) => [item.observed, item.retrieved]);
  const minY = Math.min(...values) - 0.02;
  const maxY = Math.max(...values) + 0.02;

  const project = (x: number, y: number) => {
    const px = padding + ((x - minX) / (maxX - minX)) * (width - padding * 2);
    const py = height - padding - ((y - minY) / (maxY - minY)) * (height - padding * 2);
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  };

  const line = (field: "observed" | "retrieved") =>
    spectrumSeries.map((item) => project(item.wavelength, item[field])).join(" ");

  return (
    <div className="figure-frame">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Transmission spectrum">
        <rect x="0" y="0" width={width} height={height} rx="20" fill="rgba(255,255,255,0.38)" />
        {[0, 1, 2, 3].map((step) => {
          const y = padding + step * ((height - padding * 2) / 3);
          return (
            <line
              key={step}
              x1={padding}
              x2={width - padding}
              y1={y}
              y2={y}
              stroke="rgba(17,17,17,0.08)"
              strokeDasharray="4 6"
            />
          );
        })}
        <polyline
          fill="none"
          stroke="rgba(125,69,65,0.55)"
          strokeWidth="3"
          strokeLinejoin="round"
          strokeLinecap="round"
          points={line("observed")}
        />
        <polyline
          fill="none"
          stroke="#0d5f63"
          strokeWidth="3"
          strokeLinejoin="round"
          strokeLinecap="round"
          points={line("retrieved")}
        />
        <text x={padding} y={height - 4} fontSize="12" fill="rgba(17,17,17,0.55)">
          0.5 μm
        </text>
        <text x={width - padding - 34} y={height - 4} fontSize="12" fill="rgba(17,17,17,0.55)">
          5.0 μm
        </text>
        <text x={padding} y={18} fontSize="12" fill="rgba(17,17,17,0.55)">
          normalized transit depth
        </text>
      </svg>
    </div>
  );
}
