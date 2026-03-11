import { Telescope, Cpu, FlaskConical, Orbit, Zap, BarChart3 } from "lucide-react";
import { ReactNode } from "react";

interface Feature {
  icon: ReactNode;
  title: string;
  description: string;
}

const features: Feature[] = [
  {
    icon: <Telescope className="w-5 h-5" />,
    title: "Transmission Spectra",
    description:
      "Analyze light filtered through exoplanet atmospheres to identify molecular absorption features.",
  },
  {
    icon: <FlaskConical className="w-5 h-5" />,
    title: "Biosignature Detection",
    description:
      "Classify atmospheric compositions for signs of biological activity — methane, ozone, water vapor.",
  },
  {
    icon: <Cpu className="w-5 h-5" />,
    title: "Quantum Hardware",
    description:
      "Run inference on real 5-qubit IQM Spark processors via the Odra quantum computer at PWR Wroclaw.",
  },
  {
    icon: <Orbit className="w-5 h-5" />,
    title: "16 Exoplanet Catalog",
    description:
      "Browse TRAPPIST-1 system, K2-18b, Proxima Centauri b, and more — with real JWST data where available.",
  },
  {
    icon: <Zap className="w-5 h-5" />,
    title: "QELM Architecture",
    description:
      "Quantum extreme learning machines encode spectra into quantum states for single-pass classification.",
  },
  {
    icon: <BarChart3 className="w-5 h-5" />,
    title: "Model Comparison",
    description:
      "Compare two quantum models against a classical random forest baseline across confidence, gases, and speed.",
  },
];

export function Features() {
  return (
    <section className="relative py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="reveal text-3xl sm:text-4xl font-bold mb-4">
            Everything you need for
            <span className="gradient-text"> biosignature analysis</span>
          </h2>
          <p className="reveal reveal-delay-1 text-muted max-w-xl mx-auto">
            From raw spectra to biological classification, powered by quantum computing
            and validated against classical methods.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((f, i) => (
            <div
              key={f.title}
              className={`reveal reveal-delay-${(i % 3) + 1} group p-6 rounded-xl border border-border bg-surface/50 hover:border-muted transition-all duration-300`}
            >
              <div className="w-10 h-10 rounded-lg bg-bg border border-border flex items-center justify-center text-purple mb-4 group-hover:text-pink transition-colors">
                {f.icon}
              </div>
              <h3 className="text-sm font-semibold mb-2">{f.title}</h3>
              <p className="text-xs text-muted leading-relaxed">{f.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
