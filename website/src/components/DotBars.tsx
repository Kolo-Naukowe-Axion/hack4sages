interface DotDatum {
  label: string;
  meta: string;
  value: number;
  color?: string;
  isVerified?: boolean;
}

interface DotBarsProps {
  title: string;
  note: string;
  unit: string;
  precision?: number;
  data: DotDatum[];
}

const DOT_COUNT = 26;

export function DotBars({
  title,
  note,
  unit,
  precision = 3,
  data,
}: DotBarsProps) {
  const maxValue = Math.max(...data.map((item) => item.value));

  return (
    <article className="chart-shell">
      <div className="chart-meta">
        <div>
          <p className="chart-title">{title}</p>
          <p className="chart-note">{note}</p>
        </div>
      </div>

      <div className="dot-bars">
        {data.map((item) => {
          const activeDots = Math.max(1, Math.round((item.value / maxValue) * DOT_COUNT));

          return (
            <div key={item.label} className="dot-row">
              <div className="dot-row-copy">
                <p className="dot-row-label">{item.label}</p>
                <p className="dot-row-meta">
                  {item.meta} · {item.isVerified ? "repo-verified" : "placeholder"}
                </p>
              </div>

              <div className="dot-strip" aria-hidden="true">
                {Array.from({ length: DOT_COUNT }).map((_, index) => (
                  <span
                    key={`${item.label}-${index}`}
                    className={`dot-cell ${index < activeDots ? "dot-cell-active" : ""} ${item.isVerified ? "dot-cell-verified" : ""}`}
                    style={{
                      backgroundColor:
                        index < activeDots ? (item.color ?? "#f2efe6") : undefined,
                    }}
                  />
                ))}
              </div>

              <p className="dot-row-value">
                {item.value.toFixed(precision)}
                <span>{unit}</span>
              </p>
            </div>
          );
        })}
      </div>
    </article>
  );
}
