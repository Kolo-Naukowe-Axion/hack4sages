import Link from "next/link";
import { CircuitGlyph } from "@/app/components/CircuitGlyph";
import { MetricBars } from "@/app/components/MetricBars";
import { MiniSpectrum } from "@/app/components/MiniSpectrum";
import { gases, performance, projectFacts, routeVariants, storyline } from "@/app/lib/project-data";

export default function HomePage() {
  return (
    <div className="page">
      <div className="shell">
        <section className="hero-grid">
          <article className="panel hero-panel">
            <span className="eyebrow">Scientific demo deck for a two-minute judge presentation</span>
            <h1>Quantifying Five-Gas Biosignatures from Ariel Spectra</h1>
            <p>
              This app is a design lab for presenting your hackathon project as something closer to a
              live scientific briefing than a conventional website. Every route below uses the same
              verified project facts from this repository, but frames them with a different design
              language for different judges, moods, and demo styles.
            </p>
            <div className="chip-row">
              {gases.map((gas) => (
                <span key={gas.key} className="chip">
                  <strong className="mono">{gas.key}</strong> • {gas.label}
                </span>
              ))}
            </div>
            <div className="stats-grid">
              {projectFacts.map((fact) => (
                <div key={fact.label} className="stat-card">
                  <span className="eyebrow">{fact.label}</span>
                  <strong>{fact.value}</strong>
                </div>
              ))}
            </div>
          </article>
          <div className="panel hero-panel" style={{ display: "grid", gap: "1rem" }}>
            <div className="stat-card">
              <span className="eyebrow">Best verified checkpoint</span>
              <strong className="mono">{performance.holdout.toFixed(4)} mRMSE</strong>
              <span>Holdout, re-evaluated from `best_model.pt` on {performance.rows.toLocaleString()} rows.</span>
            </div>
            <div className="stat-card">
              <span className="eyebrow">Validation during training</span>
              <strong className="mono">{performance.bestTrainingVal.toFixed(4)}</strong>
              <span>Best training-phase validation snapshot, preserved at epoch {performance.bestEpoch}.</span>
            </div>
            <div className="stat-card">
              <span className="eyebrow">Architecture</span>
              <strong>Hybrid 8-qubit residual</strong>
              <span>Classical encoder plus per-target gated quantum correction head.</span>
            </div>
          </div>
        </section>

        <section className="two-column">
          <article className="panel hero-panel">
            <h2 className="section-title">Narrative Spine</h2>
            <div className="story-grid">
              {storyline.map((step) => (
                <div key={step.title} className="story-card">
                  <strong>{step.title}</strong>
                  <p>{step.detail}</p>
                </div>
              ))}
            </div>
            <p className="footer-note">
              Scientific framing: the co-presence of H2O, CO2, CO, CH4, and NH3 is presented as a
              biosignature context signal, not as standalone proof of life.
            </p>
          </article>
          <div style={{ display: "grid", gap: "1rem" }}>
            <MiniSpectrum />
            <CircuitGlyph />
          </div>
        </section>

        <section className="two-column">
          <MetricBars />
          <article className="panel hero-panel">
            <h2 className="section-title">Eight Directions</h2>
            <div className="route-grid">
              {routeVariants.map((variant) => (
                <Link key={variant.slug} href={`/${variant.slug}`} className="route-card">
                  <small>Variant {variant.slug}</small>
                  <strong>{variant.title}</strong>
                  <span className="eyebrow">{variant.framing}</span>
                  <p>{variant.description}</p>
                </Link>
              ))}
            </div>
          </article>
        </section>
      </div>
    </div>
  );
}
