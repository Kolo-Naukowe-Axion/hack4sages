"use client";

import { useState } from "react";
import Link from "next/link";
import { planets } from "@/data/planets";
import { StarField } from "@/components/StarField";
import { PlanetField } from "@/components/PlanetField";
import { useScrollReveal } from "@/hooks/useScrollReveal";
import {
  Atom,
  Telescope,
  FlaskConical,
  Cpu,
  Globe,
  Zap,
  ArrowRight,
} from "lucide-react";

export default function HomePage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const scrollRef = useScrollReveal();

  const selected = selectedId ? planets.find((p) => p.id === selectedId) : null;

  return (
    <div ref={scrollRef} className="relative">
      {/* Hero */}
      <section className="relative min-h-[90vh] flex flex-col items-center justify-center overflow-hidden">
        <StarField />

        <div className="relative z-10 text-center px-6 mb-8 pt-20">
          <div className="reveal inline-flex items-center gap-2 bg-white/5 border border-white/10 rounded-full px-4 py-1.5 mb-6 text-xs text-white/50">
            <Atom className="w-3.5 h-3.5 text-cyan" />
            HACK-4-SAGES 2026 &middot; Team Axion
          </div>

          <h1 className="reveal text-5xl md:text-7xl font-extrabold text-white leading-tight tracking-tight mb-4">
            Exo<span className="text-cyan">Biome</span>
          </h1>

          <p className="reveal text-lg md:text-xl text-white/50 max-w-2xl mx-auto leading-relaxed">
            Quantum-enhanced biosignature detection for exoplanet atmospheres.
            <br className="hidden md:block" />
            Powered by QELM on real quantum hardware.
          </p>
        </div>

        {/* Mini Planet Field */}
        <div className="reveal relative z-10 w-full max-w-4xl mx-auto h-[340px] px-6">
          <div className="w-full h-full bg-space-800/30 border border-white/5 rounded-2xl overflow-hidden">
            <PlanetField
              planets={planets}
              selectedId={selectedId}
              onSelect={setSelectedId}
              compact
            />
          </div>
        </div>

        {selected && (
          <div className="reveal relative z-10 mt-4 bg-space-800/80 backdrop-blur-md border border-white/10 rounded-xl px-5 py-3 text-sm text-white/70 flex items-center gap-4">
            <span className="text-white font-semibold">{selected.name}</span>
            <span>{selected.eqTempK ?? "N/A"} K</span>
            <span>{selected.distanceLy} ly</span>
            {selected.inHabitableZone && (
              <span className="text-green text-xs font-medium">Habitable Zone</span>
            )}
            <Link
              href="/explorer"
              className="ml-2 text-cyan hover:text-white transition-colors flex items-center gap-1"
            >
              Explore <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        )}

        <div className="reveal relative z-10 mt-8 mb-4">
          <Link
            href="/explorer"
            className="inline-flex items-center gap-2 bg-gradient-to-r from-cyan to-green text-space-900 font-semibold px-8 py-3 rounded-xl text-sm hover:shadow-[0_0_30px_rgba(0,212,255,0.3)] hover:scale-105 transition-all"
          >
            <Telescope className="w-4 h-4" />
            Open Explorer
          </Link>
        </div>
      </section>

      {/* Explainer Sections */}
      <section className="relative max-w-5xl mx-auto px-6 py-24">
        <div className="grid md:grid-cols-3 gap-8">
          {[
            {
              icon: Globe,
              title: "Exoplanets",
              color: "text-cyan",
              bg: "bg-cyan/10",
              desc: "We analyze 16 confirmed exoplanets including TRAPPIST-1 system, K2-18b, and Proxima Centauri b. Each has real or synthetic transmission spectra ready for analysis.",
            },
            {
              icon: FlaskConical,
              title: "Biosignatures",
              color: "text-green",
              bg: "bg-green/10",
              desc: "Molecular fingerprints in a planet's atmosphere that hint at life: methane, ozone, water vapor, and carbon dioxide in specific ratios that defy abiotic explanation.",
            },
            {
              icon: Zap,
              title: "Quantum Detection",
              color: "text-amber",
              bg: "bg-amber/10",
              desc: "QELM (Quantum Extreme Learning Machine) running on IQM Spark's 5 real qubits. Quantum random features create exponentially rich feature maps for spectral classification.",
            },
          ].map(({ icon: Icon, title, color, bg, desc }) => (
            <div
              key={title}
              className="reveal bg-space-800/40 border border-white/5 rounded-2xl p-6 hover:border-white/10 transition-all"
            >
              <div className={`w-10 h-10 ${bg} rounded-xl flex items-center justify-center mb-4`}>
                <Icon className={`w-5 h-5 ${color}`} />
              </div>
              <h3 className="text-white font-bold text-lg mb-2">{title}</h3>
              <p className="text-sm text-white/40 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Stats Bar */}
      <section className="relative border-y border-white/5 py-12">
        <div className="max-w-4xl mx-auto px-6 grid grid-cols-3 gap-8 text-center">
          {[
            { value: "3", label: "Models", sublabel: "2 quantum + 1 classical", icon: Cpu },
            { value: "16", label: "Exoplanets", sublabel: "5 with JWST data", icon: Globe },
            { value: "5", label: "Qubits", sublabel: "IQM Spark real hardware", icon: Atom },
          ].map(({ value, label, sublabel, icon: Icon }) => (
            <div key={label} className="reveal">
              <Icon className="w-5 h-5 text-cyan/40 mx-auto mb-2" />
              <div className="text-3xl font-extrabold text-white mb-1">{value}</div>
              <div className="text-sm text-white/50 font-medium">{label}</div>
              <div className="text-xs text-white/25 mt-0.5">{sublabel}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
