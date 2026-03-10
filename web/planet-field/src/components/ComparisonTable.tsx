const rows = [
  { metric: "Accuracy (K2-18b)", qelm1: "94.2%", qelm2: "91.7%", rf: "96.8%" },
  { metric: "Avg. Confidence", qelm1: "78.3%", qelm2: "74.1%", rf: "82.6%" },
  { metric: "Processing Time", qelm1: "~2.3s", qelm2: "~1.8s", rf: "~0.1s" },
  { metric: "Qubits Used", qelm1: "5", qelm2: "5", rf: "N/A" },
  { metric: "Feature Encoding", qelm1: "Angle", qelm2: "Amplitude", rf: "PCA" },
  { metric: "Hardware", qelm1: "IQM Spark", qelm2: "IQM Spark", rf: "CPU" },
  { metric: "Training Samples", qelm1: "500", qelm2: "500", rf: "10,000" },
  { metric: "Unique Advantage", qelm1: "Noise-resilient", qelm2: "Richer states", rf: "Speed baseline" },
];

export function ComparisonTable() {
  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      <div className="p-5 border-b border-border">
        <h3 className="font-semibold text-text">Model Comparison</h3>
        <p className="text-sm text-muted mt-1">
          Side-by-side metrics across all three models
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left px-5 py-3 text-[11px] uppercase tracking-wider text-muted font-medium">
                Metric
              </th>
              <th className="text-left px-5 py-3 text-[11px] uppercase tracking-wider text-accent-blue font-medium">
                QELM Vetrano
              </th>
              <th className="text-left px-5 py-3 text-[11px] uppercase tracking-wider text-accent-blue font-medium">
                QELM Extended
              </th>
              <th className="text-left px-5 py-3 text-[11px] uppercase tracking-wider text-accent-green font-medium">
                Classical RF
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={row.metric}
                className={`border-b border-border ${
                  i % 2 === 0 ? "bg-surface" : "bg-elevated"
                }`}
              >
                <td className="px-5 py-2.5 text-muted font-medium">{row.metric}</td>
                <td className="px-5 py-2.5 text-text font-mono">{row.qelm1}</td>
                <td className="px-5 py-2.5 text-text font-mono">{row.qelm2}</td>
                <td className="px-5 py-2.5 text-text font-mono">{row.rf}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
