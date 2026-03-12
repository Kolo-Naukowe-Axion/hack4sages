import { performance } from "@/app/lib/project-data";

export function MetricBars() {
  const maxValue = Math.max(...performance.perGasRmse.map((entry) => entry.value));

  return (
    <div className="figure-frame">
      <h3 className="section-title">Per-Gas Error</h3>
      <div style={{ display: "grid", gap: "0.85rem" }}>
        {performance.perGasRmse.map((entry) => (
          <div key={entry.gas} style={{ display: "grid", gap: "0.35rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem" }}>
              <span className="mono">{entry.gas}</span>
              <span className="mono">{entry.value.toFixed(3)}</span>
            </div>
            <div
              aria-hidden="true"
              style={{
                height: "10px",
                borderRadius: "999px",
                background: "rgba(17, 17, 17, 0.08)",
                overflow: "hidden"
              }}
            >
              <div
                style={{
                  width: `${(entry.value / maxValue) * 100}%`,
                  height: "100%",
                  borderRadius: "999px",
                  background: "linear-gradient(90deg, #0d5f63, #d79a2b)"
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
