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
    <div className="min-h-[40vh] py-12">
      <h2 className="font-display text-2xl font-semibold text-heading">
        Model Comparison
      </h2>
      <p className="mt-2 text-muted">
        Side-by-side performance metrics across all three approaches.
      </p>

      <div className="mt-8 overflow-x-auto rounded-2xl bg-deep shadow-lg shadow-black/5">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-surface">
              <th className="px-6 py-4 text-left font-display font-semibold text-heading">
                Metric
              </th>
              <th className="border-l-2 border-cyan/20 px-6 py-4 text-center font-display font-semibold text-heading">
                QELM Vetrano
              </th>
              <th className="border-l-2 border-cyan/20 px-6 py-4 text-center font-display font-semibold text-heading">
                QELM Extended
              </th>
              <th className="px-6 py-4 text-center font-display font-semibold text-heading">
                Classical RF
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={row.metric}
                className={i % 2 === 1 ? "bg-surface/50" : ""}
              >
                <td className="px-6 py-3 text-muted">{row.metric}</td>
                <td className="border-l-2 border-cyan/20 px-6 py-3 text-center font-mono text-text">
                  {row.v1}
                </td>
                <td className="border-l-2 border-cyan/20 px-6 py-3 text-center font-mono text-text">
                  {row.v2}
                </td>
                <td className="px-6 py-3 text-center font-mono text-text">
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
