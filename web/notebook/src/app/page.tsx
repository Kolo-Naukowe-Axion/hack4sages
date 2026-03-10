"use client";

import Cell from "@/components/Cell";
import CodeBlock from "@/components/CodeBlock";
import Link from "next/link";
import { useScrollReveal } from "@/hooks/useScrollReveal";
import { Telescope, FlaskConical, Atom } from "lucide-react";

export default function HomePage() {
  const ref = useScrollReveal();

  return (
    <div ref={ref} className="max-w-[900px] mx-auto px-4 py-6 space-y-6 cell-stagger">
      {/* Title — no cell */}
      <div className="reveal space-y-3">
        <h1 className="text-3xl font-bold text-nb-text">
          ExoBiome: Quantum Biosignature Detection
        </h1>
        <p className="text-nb-muted text-sm">
          Team Axion &mdash; HACK-4-SAGES 2026 &middot; ETH Zurich COPL
        </p>
        <p className="text-nb-muted text-sm">
          March 2026 &middot; Category: Life Detection and Biosignatures
        </p>
        <hr className="border-nb-border" />
      </div>

      {/* Code cell: import */}
      <div className="reveal">
        <Cell type="code" executionCount={1}>
          <CodeBlock
            lines={[
              "import exobiome",
              "from exobiome import QuantumClassifier, SpectrumAnalyzer",
              "from exobiome.data import load_planet_catalog",
              "",
              "# Initialize the ExoBiome quantum biosignature detection pipeline",
              "exobiome.describe()",
            ]}
          />
        </Cell>
      </div>

      {/* Output — free flowing */}
      <div className="reveal space-y-3 text-[14px] px-1">
        <p>
          <strong>ExoBiome v1.0</strong> &mdash; Quantum Biosignature Detection Pipeline
        </p>
        <p className="text-nb-muted leading-relaxed">
          ExoBiome analyzes transmission spectra from exoplanet atmospheres to detect
          potential biosignatures. When a planet transits its host star, starlight
          passes through the planet&apos;s atmosphere, leaving spectral fingerprints of
          atmospheric gases. By analyzing these fingerprints with quantum-enhanced
          machine learning, we can identify molecules that may indicate biological
          activity &mdash; such as methane (CH&#x2084;), ozone (O&#x2083;), and water
          (H&#x2082;O) in specific ratios.
        </p>
        <div className="flex items-center gap-2 text-nb-muted text-[13px]">
          <Telescope size={14} className="text-nb-blue" />
          <span>Data sources: JWST (MAST Archive), MultiREx synthetic spectra, NASA PSG</span>
        </div>
      </div>

      {/* Code cell: biosignatures */}
      <div className="reveal">
        <Cell type="code" executionCount={2}>
          <CodeBlock
            lines={[
              "# What are biosignatures?",
              "exobiome.describe_biosignatures()",
            ]}
          />
        </Cell>
      </div>

      {/* Output — biosignatures explanation (free flowing) */}
      <div className="reveal space-y-3 text-[14px] px-1">
        <p>
          <strong>Biosignatures</strong> are atmospheric gases or gas combinations whose
          presence is difficult to explain without biological processes.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3">
          <div className="border border-nb-border rounded-lg p-3 bg-nb-card">
            <div className="font-mono text-[12px] text-nb-blue font-semibold mb-1">CH&#x2084; (Methane)</div>
            <p className="text-[12px] text-nb-muted">
              Produced by methanogenic archaea. Short atmospheric lifetime requires
              continuous replenishment &mdash; a strong indicator of active biology.
            </p>
          </div>
          <div className="border border-nb-border rounded-lg p-3 bg-nb-card">
            <div className="font-mono text-[12px] text-nb-green font-semibold mb-1">O&#x2083; (Ozone)</div>
            <p className="text-[12px] text-nb-muted">
              Photochemical product of O&#x2082;. On Earth, oxygen is overwhelmingly
              biogenic. Ozone serves as a proxy for atmospheric oxygen abundance.
            </p>
          </div>
          <div className="border border-nb-border rounded-lg p-3 bg-nb-card">
            <div className="font-mono text-[12px] text-nb-orange font-semibold mb-1">CH&#x2084; + CO&#x2082; (Disequilibrium)</div>
            <p className="text-[12px] text-nb-muted">
              Co-existence of reduced and oxidized gases implies thermodynamic
              disequilibrium, which biology can sustain but abiotic chemistry cannot.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-nb-muted text-[13px]">
          <FlaskConical size={14} className="text-nb-green" />
          <span>Reference: Schwieterman et al. 2018 &mdash; Exoplanet Biosignatures Review</span>
        </div>
      </div>

      {/* Code cell: quantum detection */}
      <div className="reveal">
        <Cell type="code" executionCount={3}>
          <CodeBlock
            lines={[
              "# How does quantum-enhanced detection work?",
              "exobiome.describe_quantum_detection()",
            ]}
          />
        </Cell>
      </div>

      {/* Output — quantum explanation (free flowing) */}
      <div className="reveal space-y-3 text-[14px] px-1">
        <p>
          <strong>Quantum Extreme Learning Machine (QELM)</strong> leverages quantum
          hardware to project spectral features into high-dimensional Hilbert space,
          enabling superior pattern recognition for noisy spectral data.
        </p>
        <div className="font-mono text-[12px] bg-nb-code-bg text-[#abb2bf] p-3 rounded-lg leading-relaxed">
          <div className="text-[#5c6370]"># Pipeline architecture</div>
          <div>Spectrum (wavelength, flux) &rarr; Feature Encoding &rarr; Quantum Reservoir (IQM Spark, 5 qubits)</div>
          <div>&nbsp;&nbsp;&rarr; Measurement &rarr; Classical Readout Layer &rarr; Biosignature Classification</div>
          <div className="mt-2 text-[#5c6370]"># Based on Vetrano et al. 2025 (arXiv:2509.03617)</div>
          <div className="text-[#5c6370]"># Hardware: Odra 5 (PWR Wroclaw) / VTT Q50 (Finland)</div>
        </div>
        <div className="flex items-center gap-2 text-nb-muted text-[13px]">
          <Atom size={14} className="text-[#af00db]" />
          <span>First-ever application of quantum ML to biosignature detection</span>
        </div>
      </div>

      {/* Stats — no cell */}
      <div className="reveal grid grid-cols-3 gap-4">
        <StatBlock value="3" label="Detection Models" sub="2 quantum + 1 classical" />
        <StatBlock value="16" label="Exoplanets" sub="4 with JWST data" />
        <StatBlock value="5" label="Qubits" sub="IQM Spark (Odra 5)" />
      </div>

      {/* Navigation — no cell */}
      <div className="reveal space-y-3 px-1">
        <hr className="border-nb-border" />
        <p className="text-[14px] text-nb-muted">Continue to the interactive notebooks:</p>
        <div className="flex gap-3">
          <Link
            href="/explorer"
            className="flex-1 text-center px-4 py-3 bg-nb-blue text-white rounded-lg font-mono text-[13px] hover:bg-blue-600 transition-colors"
          >
            Planet_Explorer.ipynb
          </Link>
          <Link
            href="/models"
            className="flex-1 text-center px-4 py-3 bg-nb-card text-nb-text border border-nb-border rounded-lg font-mono text-[13px] hover:bg-nb-hover transition-colors"
          >
            Model_Comparison.ipynb
          </Link>
        </div>
      </div>
    </div>
  );
}

function StatBlock({ value, label, sub }: { value: string; label: string; sub: string }) {
  return (
    <div className="text-center p-4 border border-nb-border rounded-lg bg-nb-cell">
      <div className="text-2xl font-bold font-mono text-nb-blue">{value}</div>
      <div className="text-[13px] font-semibold text-nb-text mt-1">{label}</div>
      <div className="text-[11px] text-nb-muted">{sub}</div>
    </div>
  );
}
