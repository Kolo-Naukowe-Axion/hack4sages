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
    <div className="bg-space-800/60 backdrop-blur-md border border-white/8 rounded-2xl overflow-hidden">
      <div className="p-6 border-b border-white/5">
        <h3 className="font-bold text-white text-lg">Model Comparison</h3>
        <p className="text-sm text-white/40 mt-1">
          Side-by-side metrics across all three models
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/5">
              <th className="text-left px-6 py-3 text-[10px] uppercase tracking-wider text-white/30 font-medium">
                Metric
              </th>
              <th className="text-left px-6 py-3 text-[10px] uppercase tracking-wider text-cyan/60 font-medium">
                QELM Vetrano
              </th>
              <th className="text-left px-6 py-3 text-[10px] uppercase tracking-wider text-cyan/60 font-medium">
                QELM Extended
              </th>
              <th className="text-left px-6 py-3 text-[10px] uppercase tracking-wider text-amber/60 font-medium">
                Classical RF
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={row.metric}
                className={`border-b border-white/3 ${
                  i % 2 === 0 ? "bg-white/[0.01]" : ""
                }`}
              >
                <td className="px-6 py-2.5 text-white/50 font-medium">{row.metric}</td>
                <td className="px-6 py-2.5 text-white">{row.qelm1}</td>
                <td className="px-6 py-2.5 text-white">{row.qelm2}</td>
                <td className="px-6 py-2.5 text-white">{row.rf}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
