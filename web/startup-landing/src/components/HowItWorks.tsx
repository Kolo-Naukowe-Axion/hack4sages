const steps = [
  {
    num: "01",
    title: "Input spectrum",
    description:
      "Select an exoplanet and load its transmission spectrum — real JWST data or high-fidelity synthetic spectra from the ABC database.",
  },
  {
    num: "02",
    title: "Quantum encoding",
    description:
      "The spectrum is encoded into quantum states and processed through a QELM reservoir on 5-qubit IQM hardware. Readout is fed to a trained classifier.",
  },
  {
    num: "03",
    title: "Biosignature verdict",
    description:
      "Three models return independent verdicts — detected, uncertain, or none — with confidence scores and identified atmospheric gases.",
  },
];

export function HowItWorks() {
  return (
    <section className="relative py-24 px-6">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="reveal text-3xl sm:text-4xl font-bold mb-4">
            How it <span className="gradient-text">works</span>
          </h2>
          <p className="reveal reveal-delay-1 text-muted">
            Three steps from raw photons to biosignature classification.
          </p>
        </div>

        <div className="relative">
          <div className="absolute left-[23px] top-8 bottom-8 w-px bg-gradient-to-b from-indigo via-purple to-pink opacity-30" />

          <div className="space-y-12">
            {steps.map((step, i) => (
              <div key={step.num} className={`reveal reveal-delay-${i + 1} relative flex gap-6`}>
                <div className="flex-shrink-0 w-12 h-12 rounded-xl gradient-bg flex items-center justify-center text-sm font-bold text-white z-10">
                  {step.num}
                </div>
                <div className="pt-1">
                  <h3 className="text-base font-semibold mb-2">{step.title}</h3>
                  <p className="text-sm text-muted leading-relaxed">{step.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
