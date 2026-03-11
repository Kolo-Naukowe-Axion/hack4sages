const codeLines = [
  { text: "from exobiome import QELMClassifier", color: "text-text" },
  { text: "", color: "" },
  { text: "# Load JWST transmission spectrum", color: "text-muted" },
  { text: 'spectrum = load_spectrum("K2-18b")', color: "text-text" },
  { text: "", color: "" },
  { text: "# Initialize quantum model (5-qubit IQM Spark)", color: "text-muted" },
  { text: 'model = QELMClassifier(backend="odra5", n_qubits=5)', color: "text-text" },
  { text: "", color: "" },
  { text: "# Run biosignature detection", color: "text-muted" },
  { text: "result = model.predict(spectrum)", color: "text-text" },
  { text: "", color: "" },
  { text: 'print(result.verdict)      # "detected"', color: "text-green" },
  { text: "print(result.confidence)   # 0.942", color: "text-green" },
  { text: 'print(result.gases)        # ["CH₄", "CO₂", "H₂O"]', color: "text-green" },
];

export function CodeSnippet() {
  return (
    <section className="relative py-24 px-6">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="reveal text-3xl sm:text-4xl font-bold mb-4">
            Simple <span className="gradient-text">API</span>
          </h2>
          <p className="reveal reveal-delay-1 text-muted">
            Spectrum in, biosignature verdict out. Built for researchers.
          </p>
        </div>

        <div className="reveal reveal-delay-2 gradient-border rounded-xl overflow-hidden">
          <div className="bg-surface rounded-xl">
            <div className="flex items-center gap-2 px-4 py-3 border-b border-border">
              <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
              <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
              <div className="w-3 h-3 rounded-full bg-[#28c840]" />
              <span className="ml-3 text-xs text-muted font-mono">classify.py</span>
            </div>
            <div className="p-5 font-mono text-sm leading-7 overflow-x-auto">
              {codeLines.map((line, i) => (
                <div key={i} className={line.color || "h-5"}>
                  {line.text && (
                    <>
                      <span className="text-muted/40 select-none mr-6 inline-block w-5 text-right">
                        {i + 1}
                      </span>
                      <span>{line.text}</span>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
