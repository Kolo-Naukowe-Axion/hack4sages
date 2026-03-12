import type { Series } from "@/lib/types";

interface DotPlotProps {
  title: string;
  note: string;
  series: Series[];
  xLabel: string;
  yLabel: string;
}

function bounds(values: number[]) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min || 1) * 0.18;

  return { min: min - pad, max: max + pad };
}

export function DotPlot({ title, note, series, xLabel, yLabel }: DotPlotProps) {
  const allX = series.flatMap((item) => item.points.map((point) => point.x));
  const allY = series.flatMap((item) => item.points.map((point) => point.y));
  const xRange = bounds(allX);
  const yRange = bounds(allY);

  const xToSvg = (value: number) => 60 + ((value - xRange.min) / (xRange.max - xRange.min)) * 760;
  const yToSvg = (value: number) => 210 - ((value - yRange.min) / (yRange.max - yRange.min)) * 150;

  return (
    <article className="chart-shell">
      <div className="chart-meta">
        <div>
          <p className="chart-title">{title}</p>
          <p className="chart-note">{note}</p>
        </div>
      </div>

      <svg viewBox="0 0 880 260" className="chart-svg">
        {[0, 1, 2, 3, 4].map((row) => {
          const y = 40 + row * 38;

          return (
            <g key={row}>
              {Array.from({ length: 30 }).map((_, col) => (
                <rect
                  key={`${row}-${col}`}
                  x={58 + col * 26}
                  y={y - 2}
                  width="4"
                  height="4"
                  fill="rgba(17, 15, 11, 0.16)"
                />
              ))}
            </g>
          );
        })}

        {series.map((item) => {
          const points = item.points.map((point) => ({
            cx: xToSvg(point.x),
            cy: yToSvg(point.y),
          }));

          return (
            <g key={item.id}>
              {points.map((point, index) => {
                const next = points[index + 1];

                return next ? (
                  <line
                    key={`${item.id}-line-${index}`}
                    x1={point.cx}
                    y1={point.cy}
                    x2={next.cx}
                    y2={next.cy}
                    stroke={item.color}
                    strokeWidth="1.25"
                    strokeDasharray={item.isVerified ? "0" : "4 6"}
                    opacity={item.isVerified ? 0.45 : 0.22}
                  />
                ) : null;
              })}

              {points.map((point, index) => (
                <rect
                  key={`${item.id}-${index}`}
                  x={point.cx - (item.isVerified ? 4 : 3.5)}
                  y={point.cy - (item.isVerified ? 4 : 3.5)}
                  width={item.isVerified ? 8 : 7}
                  height={item.isVerified ? 8 : 7}
                  fill={item.color}
                  opacity={item.isVerified ? 1 : 0.65}
                />
              ))}
            </g>
          );
        })}

        <text x="22" y="26" className="chart-axis-label">
          {yLabel}
        </text>
        <text x="820" y="245" textAnchor="end" className="chart-axis-label">
          {xLabel}
        </text>
      </svg>

      <div className="plot-legend">
        {series.map((item) => (
          <span key={item.id} className="plot-legend-item">
            <span className="plot-legend-dot" style={{ backgroundColor: item.color, opacity: item.isVerified ? 1 : 0.65 }} />
            {item.label}
            <span className="plot-legend-meta">{item.isVerified ? "verified" : "placeholder"}</span>
          </span>
        ))}
      </div>
    </article>
  );
}
