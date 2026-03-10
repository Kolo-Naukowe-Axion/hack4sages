const rows = [
  { metric: "Accuracy", v1: "94.2%", v2: "91.7%", v3: "96.8%" },
  { metric: "Precision", v1: "92.8%", v2: "90.3%", v3: "95.4%" },
  { metric: "Recall", v1: "95.6%", v2: "93.1%", v3: "98.1%" },
  { metric: "F1 Score", v1: "94.2%", v2: "91.7%", v3: "96.7%" },
  { metric: "Training Time", v1: "~45 min", v2: "~30 min", v3: "~5 min" },
  { metric: "Inference Time", v1: "2.3s", v2: "1.8s", v3: "0.12s" },
  { metric: "Hardware", v1: "Odra 5", v2: "VTT Q50", v3: "GPU" },
  { metric: "Qubits", v1: "5", v2: "53", v3: "—" },
];

export default function ComparisonTable() {
  return (
    <div className="py-16">
      <h2 className="font-display text-2xl font-semibold tracking-tight text-heading">
        Side-by-Side Comparison
      </h2>
      <p className="mt-2 text-sm text-muted">
        Performance metrics across all three approaches.
      </p>

      <div className="mt-8 overflow-x-auto rounded-2xl bg-deep shadow-sm">
        <table className="w-full text-sm">
          <caption className="sr-only">Model performance comparison</caption>
          <thead>
            <tr className="border-b border-border/60">
              <th className="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider text-muted">
                Metric
              </th>
              <th className="px-6 py-4 text-center text-xs font-medium uppercase tracking-wider text-muted">
                QELM Vetrano
              </th>
              <th className="px-6 py-4 text-center text-xs font-medium uppercase tracking-wider text-muted">
                QELM Extended
              </th>
              <th className="px-6 py-4 text-center text-xs font-medium uppercase tracking-wider text-muted">
                Classical RF
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={row.metric}
                className={`border-b border-border/30 last:border-0 ${
                  i % 2 === 0 ? "" : "bg-surface/40"
                }`}
              >
                <td className="px-6 py-3.5 text-sm text-muted">{row.metric}</td>
                <td className="px-6 py-3.5 text-center font-mono text-sm text-heading">
                  {row.v1}
                </td>
                <td className="px-6 py-3.5 text-center font-mono text-sm text-heading">
                  {row.v2}
                </td>
                <td className="px-6 py-3.5 text-center font-mono text-sm text-heading">
                  {row.v3}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
