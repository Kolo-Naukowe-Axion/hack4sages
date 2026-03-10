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
    <div className="py-10">
      <p className="figure-caption mb-4">
        <strong>Table 2.</strong> Performance comparison across three biosignature detection models. Metrics computed on held-out test set (n = 500).
      </p>

      <div className="overflow-x-auto border border-border bg-paper">
        <table className="w-full text-sm">
          <caption className="sr-only">Model performance comparison</caption>
          <thead>
            <tr className="border-b-2 border-heading">
              <th className="px-5 py-3 text-left font-serif font-semibold text-heading">
                Metric
              </th>
              <th className="px-5 py-3 text-center font-serif font-semibold text-heading">
                QELM Vetrano
              </th>
              <th className="px-5 py-3 text-center font-serif font-semibold text-heading">
                QELM Extended
              </th>
              <th className="px-5 py-3 text-center font-serif font-semibold text-heading">
                Classical RF
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={row.metric}
                className={`border-b border-border-light ${i % 2 === 1 ? "bg-surface/40" : ""}`}
              >
                <td className="px-5 py-2.5 font-sans text-muted">{row.metric}</td>
                <td className="px-5 py-2.5 text-center font-mono text-text">
                  {row.v1}
                </td>
                <td className="px-5 py-2.5 text-center font-mono text-text">
                  {row.v2}
                </td>
                <td className="px-5 py-2.5 text-center font-mono text-text">
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
