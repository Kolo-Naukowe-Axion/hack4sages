"use client";

import Link from "next/link";
import { ArrowRight, ChevronDown } from "lucide-react";
import { useScrollReveal } from "@/hooks/useScrollReveal";

export default function Home() {
  const explainerRef = useScrollReveal();
  const statsRef = useScrollReveal();

  return (
    <>
      {/* Hero */}
      <section className="relative flex min-h-[100dvh] items-center overflow-hidden">
        {/* Gradient orbs */}
        <div
          className="hero-orb"
          style={{
            top: "15%",
            right: "5%",
            width: "550px",
            height: "550px",
            background:
              "radial-gradient(circle, rgba(0,113,227,0.12) 0%, rgba(0,113,227,0.04) 45%, transparent 70%)",
          }}
        />
        <div
          className="hero-orb-2"
          style={{
            top: "45%",
            right: "20%",
            width: "380px",
            height: "380px",
            background:
              "radial-gradient(circle, rgba(88,86,214,0.1) 0%, rgba(175,82,222,0.04) 45%, transparent 70%)",
          }}
        />

        <div className="relative mx-auto w-full max-w-7xl px-6 lg:px-8">
          <div className="hero-stagger max-w-2xl">
            <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted">
              Quantum Biosignature Detection
            </p>
            <h1 className="mt-6 font-display text-5xl font-medium leading-[1.08] tracking-tight text-heading sm:text-6xl lg:text-7xl">
              Detect Life
              <br />
              Beyond Earth.
            </h1>
            <p className="mt-6 max-w-lg text-lg leading-relaxed text-muted">
              The first quantum machine learning system for biosignature
              detection in exoplanet atmospheres.
            </p>
            <Link
              href="/explorer"
              className="mt-10 inline-flex items-center gap-2.5 rounded-full bg-heading px-8 py-4 text-base font-medium text-white shadow-lg shadow-heading/10 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-heading/15 active:translate-y-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-heading/30 focus-visible:ring-offset-2"
            >
              Explore Planets
              <ArrowRight size={16} />
            </Link>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2">
          <ChevronDown size={20} className="scroll-indicator text-muted/40" aria-hidden="true" />
        </div>
      </section>

      {/* Science Explainer */}
      <section ref={explainerRef} className="px-6 py-28 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="grid gap-5 lg:grid-cols-5">
            {/* Large card */}
            <div className="reveal lg:col-span-3">
              <div className="h-full rounded-2xl bg-deep p-8 shadow-sm transition-shadow duration-300 hover:shadow-md lg:p-10">
                <p className="text-xs font-medium uppercase tracking-widest text-muted">
                  Foundation
                </p>
                <h3 className="mt-3 font-display text-2xl font-semibold tracking-tight text-heading">
                  Exoplanets & Transmission Spectra
                </h3>
                <p className="mt-4 max-w-lg text-sm leading-relaxed text-muted">
                  Over 6,000 planets have been confirmed orbiting other stars.
                  When one transits its host star, starlight filters through its
                  atmosphere — each gas absorbing at characteristic wavelengths.
                  The resulting spectrum is a chemical fingerprint.
                </p>

                {/* Spectrum illustration */}
                <div className="mt-8 rounded-xl bg-surface p-5">
                  <div className="flex w-full items-end justify-between gap-[2px]">
                    {Array.from({ length: 50 }).map((_, i) => {
                      const h =
                        40 +
                        Math.sin(i * 0.35) * 15 +
                        (i > 15 && i < 20 ? -25 : 0) +
                        (i > 32 && i < 37 ? -20 : 0);
                      return (
                        <div
                          key={i}
                          className="flex-1 rounded-t-sm bg-heading/10"
                          style={{ height: `${Math.max(6, h)}px` }}
                        />
                      );
                    })}
                  </div>
                </div>
                <p className="mt-2 text-xs text-muted/60">
                  Simulated transmission spectrum with absorption dips at H₂O
                  and CH₄ wavelengths
                </p>
              </div>
            </div>

            {/* Right column */}
            <div className="flex flex-col gap-5 lg:col-span-2">
              <div className="reveal flex-1" style={{ transitionDelay: "100ms" }}>
                <div className="flex h-full flex-col rounded-2xl bg-deep p-8 shadow-sm transition-shadow duration-300 hover:shadow-md">
                  <p className="text-xs font-medium uppercase tracking-widest text-green">
                    Target
                  </p>
                  <h3 className="mt-3 font-display text-xl font-semibold text-heading">
                    Biosignatures
                  </h3>
                  <p className="mt-3 text-sm leading-relaxed text-muted">
                    Certain gas combinations — methane with ozone, oxygen with a
                    reducing gas — are difficult to explain without biology.
                    These are the strongest remote evidence for extraterrestrial
                    life.
                  </p>
                </div>
              </div>

              <div className="reveal flex-1" style={{ transitionDelay: "200ms" }}>
                <div className="flex h-full flex-col rounded-2xl bg-deep p-8 shadow-sm transition-shadow duration-300 hover:shadow-md">
                  <p className="text-xs font-medium uppercase tracking-widest text-cyan">
                    Method
                  </p>
                  <h3 className="mt-3 font-display text-xl font-semibold text-heading">
                    Quantum Detection
                  </h3>
                  <p className="mt-3 text-sm leading-relaxed text-muted">
                    Quantum extreme learning machines running on real 5-qubit
                    hardware classify whether a spectrum contains biosignature
                    patterns — pushing quantum advantage in astrophysics.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section ref={statsRef} className="px-6 py-20 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="border-t border-border/60 pt-16">
            <div className="flex flex-col gap-16 sm:flex-row sm:justify-between">
              {[
                { value: "3", label: "Models Compared" },
                { value: "16", label: "Exoplanets Analyzed" },
                { value: "5", label: "Qubit Quantum Hardware" },
              ].map(({ value, label }, i) => (
                <div
                  key={label}
                  className="reveal"
                  style={{ transitionDelay: `${i * 120}ms` }}
                >
                  <span className="font-display text-6xl font-medium tracking-tight text-heading">
                    {value}
                  </span>
                  <p className="mt-2 text-sm text-muted">{label}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
