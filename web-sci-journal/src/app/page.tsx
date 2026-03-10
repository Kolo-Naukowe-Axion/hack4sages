"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { useScrollReveal } from "@/hooks/useScrollReveal";

export default function Home() {
  const bodyRef = useScrollReveal();
  const statsRef = useScrollReveal();

  return (
    <article className="mx-auto max-w-4xl px-6 py-16 lg:py-24">
      {/* Title Block */}
      <header className="journal-stagger">
        <p className="font-sans text-xs font-medium uppercase tracking-[0.2em] text-muted">
          Research Article
        </p>
        <h1 className="mt-4 font-serif text-4xl font-bold leading-tight text-heading lg:text-5xl">
          Quantum-Enhanced Detection of Atmospheric Biosignatures in Exoplanetary Spectra
        </h1>

        <div className="mt-5 flex flex-wrap items-center gap-x-1.5 text-sm text-muted">
          <span>M. Szczesny,</span>
          <span>et al.</span>
          <span className="mx-2 text-border">|</span>
          <span>HACK-4-SAGES 2026</span>
          <span className="mx-2 text-border">|</span>
          <span>ETH Zurich COPL</span>
        </div>

        <hr className="journal-rule-double mt-6" />
      </header>

      {/* Abstract */}
      <section className="journal-stagger mt-10">
        <h2 className="font-serif text-sm font-bold uppercase tracking-wider text-heading">
          Abstract
        </h2>
        <p className="mt-3 text-text">
          We present <em>ExoBiome</em>, a comparative framework for detecting atmospheric
          biosignatures in exoplanetary transmission spectra. Using quantum extreme
          learning machines (QELMs) running on real 5-qubit and 53-qubit quantum hardware,
          we classify whether observed or synthetic spectra contain gas combinations indicative
          of biological activity. Our approach is benchmarked against an optimized classical
          Random Forest baseline across a catalog of 16 exoplanets. Results demonstrate that
          quantum models achieve competitive classification accuracy (91.7–94.2%) relative
          to classical methods (96.8%), while operating on fundamentally different computational
          substrates — suggesting a pathway for quantum advantage as hardware scales.
        </p>
      </section>

      <hr className="journal-rule mt-10" />

      {/* Body Sections */}
      <div ref={bodyRef}>
        <section className="reveal mt-10 opacity-0 translate-y-4 transition-all duration-700">
          <p className="section-number text-sm">§1</p>
          <h2 className="font-serif text-2xl font-semibold text-heading">
            Introduction
          </h2>
          <div className="mt-4 space-y-4 text-text">
            <p>
              Over 6,000 exoplanets have been confirmed to date, many in the
              habitable zones of their host stars. When a planet transits its star,
              starlight filters through the planetary atmosphere, and each gas absorbs
              at characteristic wavelengths — producing a transmission spectrum that
              serves as a chemical fingerprint of the alien atmosphere.
            </p>
            <p>
              Certain gas combinations are difficult to explain without biology:
              methane coexisting with ozone, or oxygen alongside reducing gases.
              These <em>biosignatures</em> represent the strongest remote evidence
              we can gather for extraterrestrial life. However, reliably detecting
              these faint spectral features requires sophisticated classification
              methods capable of operating at the noise floor of current instruments.
            </p>
          </div>
        </section>

        <section className="reveal mt-10 opacity-0 translate-y-4 transition-all duration-700">
          <p className="section-number text-sm">§2</p>
          <h2 className="font-serif text-2xl font-semibold text-heading">
            Approach
          </h2>
          <div className="mt-4 space-y-4 text-text">
            <p>
              We employ quantum extreme learning machines — random quantum reservoirs
              where input spectra are angle-encoded into qubit rotations, processed
              through fixed entangling circuits, and measured in the computational basis.
              The resulting measurement statistics are fed to a classical linear layer
              (SVD-based) for final classification (see{" "}
              <Link href="/models" className="text-accent underline decoration-accent/30 hover:decoration-accent">
                §5 Methods
              </Link>
              ).
            </p>
            <p>
              Two quantum architectures are compared: a faithful reproduction of
              Vetrano et al. (2025) on IQM Spark hardware [1], and our modified
              topology with an extended entanglement pattern. Both are benchmarked
              against a hyperparameter-optimized Random Forest. Analysis is performed
              across 16 exoplanets spanning confirmed JWST targets and synthetic spectra (see{" "}
              <Link href="/explorer" className="text-accent underline decoration-accent/30 hover:decoration-accent">
                §3 Explorer
              </Link>
              ).
            </p>
          </div>
        </section>
      </div>

      <hr className="journal-rule mt-10" />

      {/* Key Figures */}
      <section ref={statsRef} className="mt-10">
        <div className="reveal grid grid-cols-3 gap-6 border border-border bg-paper p-6 opacity-0 translate-y-4 transition-all duration-700">
          {[
            { value: "3", label: "Models compared" },
            { value: "16", label: "Exoplanets analyzed" },
            { value: "5", label: "Qubit hardware" },
          ].map(({ value, label }) => (
            <div key={label} className="text-center">
              <span className="font-mono text-3xl font-bold text-heading">
                {value}
              </span>
              <p className="mt-1 font-sans text-xs uppercase tracking-wide text-muted">
                {label}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <div className="mt-12 text-center">
        <Link
          href="/explorer"
          className="inline-flex items-center gap-2 border border-accent bg-accent px-6 py-3 font-serif text-sm font-semibold text-white transition-colors hover:bg-accent-light"
        >
          Begin Analysis
          <ArrowRight size={16} />
        </Link>
        <p className="mt-3 font-sans text-xs text-muted">
          Select an exoplanet and run biosignature detection
        </p>
      </div>
    </article>
  );
}
