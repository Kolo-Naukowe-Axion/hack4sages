"use client";

import { useState } from "react";
import Link from "next/link";
import { planets } from "@/data/planets";
import { PlanetField } from "@/components/PlanetField";
import { useScrollReveal } from "@/hooks/useScrollReveal";
import { Telescope, FlaskConical, Zap, ArrowRight } from "lucide-react";

export default function HomePage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const scrollRef = useScrollReveal();

  return (
    <div ref={scrollRef} className="page-enter">
      {/* Hero with mini planet field */}
      <section className="relative h-screen min-h-[600px] overflow-hidden">
        <div className="absolute inset-0">
          <PlanetField
            planets={planets}
            selectedId={selectedId}
            onSelect={setSelectedId}
            compact
          />
        </div>

        <div className="relative z-10 flex flex-col items-center justify-center h-full px-6 pointer-events-none">
          <h1 className="reveal text-5xl md:text-7xl font-bold text-text leading-tight tracking-tight mb-4 text-center">
            ExoBiome
          </h1>
          <p className="reveal text-lg md:text-xl text-muted max-w-2xl mx-auto leading-relaxed mb-10 text-center">
            Quantum biosignature detection in exoplanet atmospheres
          </p>

          <div className="reveal pointer-events-auto">
            <Link
              href="/explorer"
              className="inline-flex items-center gap-2 bg-accent-blue text-white font-medium px-6 py-2.5 rounded-md text-sm
                hover:bg-accent-blue/80 hover:-translate-y-0.5 transition-all duration-200"
            >
              Explore Planets
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>

        <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-bg to-transparent z-10 pointer-events-none" />
      </section>

      {/* Info Cards */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 py-16">
        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              icon: Telescope,
              title: "Transmission Spectra",
              desc: "We analyze transmission spectra of 16 confirmed exoplanets including TRAPPIST-1 system, K2-18b, and Proxima Centauri b. Each has real JWST or synthetic spectral data.",
            },
            {
              icon: FlaskConical,
              title: "Biosignatures",
              desc: "Molecular fingerprints in a planet's atmosphere that hint at life: methane, ozone, water vapor, and carbon dioxide in specific ratios that defy abiotic explanation.",
            },
            {
              icon: Zap,
              title: "Quantum Detection",
              desc: "QELM (Quantum Extreme Learning Machine) running on IQM Spark's 5 real qubits. Quantum random features create exponentially rich feature maps for spectral classification.",
            },
          ].map(({ icon: Icon, title, desc }) => (
            <div
              key={title}
              className="reveal bg-surface border border-border rounded-lg p-5
                hover:border-muted/30 hover:-translate-y-1 transition-all duration-300"
            >
              <Icon className="w-5 h-5 text-muted mb-3" />
              <h3 className="text-text font-semibold mb-2">{title}</h3>
              <p className="text-sm text-muted leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Stats */}
      <section className="relative z-10 border-y border-border py-12">
        <div className="max-w-4xl mx-auto px-6 grid grid-cols-3 gap-8 text-center">
          {[
            { value: "3", label: "Models Compared" },
            { value: "16", label: "Exoplanets Analyzed" },
            { value: "5", label: "Qubit Hardware" },
          ].map(({ value, label }) => (
            <div key={label} className="reveal">
              <div className="text-3xl font-bold font-mono text-text mb-1">
                {value}
              </div>
              <div className="text-sm text-muted">{label}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
