"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import { ArrowRight, Globe, Dna, Atom } from "lucide-react";

function useScrollReveal() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("animate-in");
          }
        });
      },
      { threshold: 0.1 }
    );

    const children = el.querySelectorAll(".reveal");
    children.forEach((child) => observer.observe(child));

    return () => observer.disconnect();
  }, []);

  return ref;
}

export default function Home() {
  const explainerRef = useScrollReveal();
  const statsRef = useScrollReveal();

  return (
    <>
      {/* Hero */}
      <section className="relative min-h-screen flex items-center overflow-hidden">
        {/* Editorial accent — thin ruled line */}
        <div className="pointer-events-none absolute top-[20%] right-0 w-[40%] h-px bg-border" />
        <div className="pointer-events-none absolute top-[22%] right-0 w-[35%] h-px bg-border/50" />

        <div className="relative mx-auto w-full max-w-7xl px-6 lg:px-8">
          <div className="max-w-2xl">
            <p className="font-mono text-sm uppercase tracking-widest text-muted">
              Quantum Biosignature Detection
            </p>
            <h1 className="mt-4 font-display text-5xl font-semibold leading-[1.1] tracking-tight text-heading lg:text-7xl">
              Detect Life
              <br />
              Beyond Earth
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted lg:text-xl">
              The first quantum machine learning system for biosignature
              detection in exoplanet atmospheres.
            </p>
            <Link
              href="/explorer"
              className="mt-10 inline-flex items-center gap-2 rounded-full bg-cyan px-8 py-4 font-display text-base font-semibold text-white shadow-lg shadow-cyan/20 transition-all hover:shadow-xl hover:shadow-cyan/30 hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-cyan/30"
            >
              Explore Planets
              <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>

      {/* Science Explainer */}
      <section
        ref={explainerRef}
        className="min-h-[60vh] px-6 py-24 lg:px-8"
      >
        <div className="mx-auto max-w-7xl">
          <div className="grid gap-6 lg:grid-cols-5">
            {/* Left — large card */}
            <div className="reveal opacity-0 translate-y-6 transition-all duration-700 lg:col-span-3">
              <div className="rounded-2xl bg-deep p-8 shadow-lg shadow-black/5 transition-all duration-300 hover:shadow-xl hover:shadow-black/10 lg:p-10">
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-cyan/10 text-cyan">
                    <Globe size={24} />
                  </div>
                  <div>
                    <h3 className="font-display text-xl font-semibold text-heading">
                      Exoplanets & Transmission Spectra
                    </h3>
                    <p className="mt-3 leading-relaxed text-muted">
                      Over 6,000 planets have been confirmed orbiting other
                      stars. When one of these exoplanets transits its host
                      star, starlight filters through its atmosphere — each gas
                      absorbing at characteristic wavelengths. The resulting
                      transmission spectrum is a chemical fingerprint of that
                      alien atmosphere.
                    </p>
                  </div>
                </div>

                {/* Inline spectrum illustration */}
                <div className="mt-8 flex items-end gap-px rounded-xl bg-surface p-6">
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
                          className="flex-1 rounded-t-sm bg-cyan/30"
                          style={{ height: `${Math.max(6, h)}px` }}
                        />
                      );
                    })}
                  </div>
                </div>
                <p className="mt-2 text-xs text-muted">
                  Simulated transmission spectrum with absorption dips at H₂O
                  and CH₄ wavelengths
                </p>
              </div>
            </div>

            {/* Right — two stacked cards */}
            <div className="flex flex-col gap-6 lg:col-span-2">
              <div className="reveal opacity-0 translate-y-6 transition-all duration-700 delay-150 flex-1">
                <div className="h-full rounded-2xl bg-deep p-8 shadow-lg shadow-black/5 transition-all duration-300 hover:shadow-xl hover:shadow-black/10">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-green/10 text-green">
                    <Dna size={24} />
                  </div>
                  <h3 className="mt-4 font-display text-xl font-semibold text-heading">
                    Biosignatures
                  </h3>
                  <p className="mt-3 leading-relaxed text-muted">
                    Certain gas combinations — like methane with ozone, or
                    oxygen with a reducing gas — are difficult to explain
                    without biology. These biosignatures are the strongest
                    remote evidence we can gather for extraterrestrial life.
                  </p>
                </div>
              </div>

              <div className="reveal opacity-0 translate-y-6 transition-all duration-700 delay-300 flex-1">
                <div className="h-full rounded-2xl bg-deep p-8 shadow-lg shadow-black/5 transition-all duration-300 hover:shadow-xl hover:shadow-black/10">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-teal/10 text-teal">
                    <Atom size={24} />
                  </div>
                  <h3 className="mt-4 font-display text-xl font-semibold text-heading">
                    Quantum Detection
                  </h3>
                  <p className="mt-3 leading-relaxed text-muted">
                    We use quantum extreme learning machines running on real
                    5-qubit hardware to classify whether a spectrum contains
                    biosignature patterns — pushing the frontier of quantum
                    advantage in astrophysics.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Separator */}
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <hr className="h-px border-0 bg-border" />
      </div>

      {/* Stats */}
      <section ref={statsRef} className="px-6 py-20 lg:px-8">
        <div className="mx-auto flex max-w-7xl flex-col gap-12 sm:flex-row sm:justify-between">
          {[
            { value: "3", label: "Models Compared" },
            { value: "16", label: "Exoplanets Analyzed" },
            { value: "5", label: "Qubit Quantum Hardware" },
          ].map(({ value, label }, i) => (
            <div
              key={label}
              className={`reveal opacity-0 translate-y-6 transition-all duration-700`}
              style={{ transitionDelay: `${i * 150}ms` }}
            >
              <span className="font-display text-5xl font-semibold text-heading">
                {value}
              </span>
              <p className="mt-2 text-sm uppercase tracking-wide text-muted">
                {label}
              </p>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
