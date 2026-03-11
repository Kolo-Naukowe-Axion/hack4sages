const stats = [
  { value: "3", label: "ML Models", detail: "2 quantum + 1 classical" },
  { value: "16", label: "Exoplanets", detail: "Including JWST targets" },
  { value: "5", label: "Qubit Hardware", detail: "IQM Spark processor" },
];

export function Stats() {
  return (
    <section className="relative py-24 px-6">
      <div className="max-w-4xl mx-auto">
        <div className="grid grid-cols-3 gap-8">
          {stats.map((s, i) => (
            <div key={s.label} className={`reveal reveal-delay-${i + 1} text-center`}>
              <div className="text-5xl sm:text-6xl font-extrabold gradient-text mb-2">
                {s.value}
              </div>
              <div className="text-sm font-semibold text-text mb-1">{s.label}</div>
              <div className="text-xs text-muted">{s.detail}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
