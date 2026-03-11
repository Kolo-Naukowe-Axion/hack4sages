import Link from "next/link";
import { ArrowRight } from "lucide-react";

export function Hero() {
  return (
    <section className="relative pt-40 pb-24 px-6">
      <div className="max-w-4xl mx-auto text-center">
        <div className="reveal">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-border text-xs text-muted mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-green" />
            Built for HACK-4-SAGES 2026
          </div>
        </div>

        <h1 className="reveal reveal-delay-1 text-5xl sm:text-7xl font-extrabold tracking-tight leading-[1.05] mb-6">
          Detect life beyond Earth
          <br />
          <span className="gradient-text">with quantum ML</span>
        </h1>

        <p className="reveal reveal-delay-2 text-lg sm:text-xl text-muted max-w-2xl mx-auto mb-10 leading-relaxed">
          ExoBiome analyzes exoplanet transmission spectra using quantum extreme learning
          machines on real 5-qubit hardware to classify atmospheric biosignatures.
        </p>

        <div className="reveal reveal-delay-3 flex items-center justify-center gap-4">
          <Link
            href="/explorer"
            className="gradient-border inline-flex items-center gap-2 px-6 py-3 text-sm font-medium rounded-xl hover:bg-surface transition-colors"
          >
            Explore planets
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            href="/models"
            className="inline-flex items-center gap-2 px-6 py-3 text-sm font-medium text-muted rounded-xl border border-border hover:text-text hover:border-muted transition-colors"
          >
            View models
          </Link>
        </div>
      </div>
    </section>
  );
}
