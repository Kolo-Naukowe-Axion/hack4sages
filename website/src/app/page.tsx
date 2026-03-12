import Link from "next/link";
import { DotBars } from "@/components/DotBars";
import { DotPlot } from "@/components/DotPlot";
import { DotSpectrum } from "@/components/DotSpectrum";
import { GhostSection } from "@/components/GhostSection";
import { PointCloudPlanet } from "@/components/PointCloudPlanet";
import {
  benchmarkNotes,
  heroStats,
  modelResults,
  perGasMetrics,
  pipelineSteps,
  sectionLinks,
  trainingDataTrend,
  validationCurve,
} from "@/data/presentation-data";

const chartColors = ["#726a60", "#938a7e", "#b7ad9d", "#1a1712"];

export default function HomePage() {
  return (
    <main className="paper-page">
      <section id="intro" className="hero-section section-anchor">
        <div className="hero-header">
          <p className="hero-brand">ExoBiome</p>
          <h1 className="hero-title">ExoBiome</h1>
          <p className="hero-subtitle">A deep dive on biosignature retrieval from transmission spectroscopy</p>

          <nav className="hero-tabs" aria-label="Sections">
            {sectionLinks.map((section, index) => (
              <Link
                key={section.id}
                href={`#${section.id}`}
                className={`hero-tab ${index === 0 ? "hero-tab-active" : ""}`}
              >
                {index === 0 ? "Overview" : section.label}
              </Link>
            ))}
          </nav>
        </div>

        <PointCloudPlanet />

        <div className="hero-bottom">
          <p className="scroll-note">↓ Scroll for more</p>

          <div className="hero-facts">
            {heroStats.map((stat) => (
              <div key={stat.label} className="hero-fact">
                <span>{stat.label}</span>
                <strong>{stat.value}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>

      <GhostSection
        id="problem"
        index="01"
        eyebrow="Framing"
        title="The project is about a five-gas atmospheric pattern, not a single molecule headline."
        summary="Transmission spectra need to be interpreted as a retrieval problem. The model has to estimate H2O, CO2, CO, CH4, and NH3 from weak signals plus auxiliary context well enough to support biosignature reasoning."
      >
        <div className="paper-grid paper-grid-3">
          <article className="paper-copy">
            <p className="paper-copy-label">Input regime</p>
            <p>
              Ariel-style transmission spectra compress atmospheric structure into subtle wavelength changes shaped by
              noise, instrument width, and planet-star metadata.
            </p>
          </article>
          <article className="paper-copy">
            <p className="paper-copy-label">Scientific goal</p>
            <p>
              Estimate five atmospheric abundances jointly so the downstream interpretation step can reason about a
              biosignature-relevant chemical state rather than a single detection.
            </p>
          </article>
          <article className="paper-copy">
            <p className="paper-copy-label">Pitch constraint</p>
            <p>
              The website needs to behave like a presentation artifact for judges: legible, calm, and immediately clear
              about what is verified versus provisional.
            </p>
          </article>
        </div>
      </GhostSection>

      <GhostSection
        id="pipeline"
        index="02"
        eyebrow="System"
        title="The pipeline stays simple enough to explain live: spectra in, context in, five gases out."
        summary="The page only exposes the pieces required to justify the result in a two-minute presentation."
      >
        <div className="paper-grid paper-grid-2">
          <article className="paper-copy paper-copy-strong">
            <p className="paper-copy-label">Model family</p>
            <p>
              We compare the ADC2023 baseline CNN, a Random Forest baseline, a state-of-the-art normalizing flow, and
              our hybrid quantum regressor with a gated residual quantum correction.
            </p>
          </article>

          <DotSpectrum />
        </div>

        <div className="pipeline-list">
          {pipelineSteps.map((step, index) => (
            <article key={step.label} className="pipeline-item">
              <p className="pipeline-item-index">{String(index + 1).padStart(2, "0")}</p>
              <h3>{step.label}</h3>
              <p>{step.detail}</p>
            </article>
          ))}
        </div>
      </GhostSection>

      <GhostSection
        id="benchmarks"
        index="03"
        eyebrow="Benchmarks"
        title="The comparison surfaces are rendered as evidence panels, not product analytics widgets."
        summary="Verified quantum results come directly from repo artifacts. Cross-model comparisons are kept visible but marked as placeholders until the final benchmark tables land."
      >
        <div className="paper-grid paper-grid-2">
          <DotBars
            title="mRMSE comparison"
            note="Quantum row is verified from repo artifacts. Remaining rows are placeholders."
            unit=""
            precision={3}
            data={modelResults.map((model, index) => ({
              label: model.label,
              meta: model.family,
              value: model.rmse,
              color: chartColors[index],
              isVerified: model.isVerified,
            }))}
          />

          <DotBars
            title="Training time"
            note="Placeholder wall-clock comparison for live presentation use."
            unit="h"
            precision={1}
            data={modelResults.map((model, index) => ({
              label: model.label,
              meta: model.family,
              value: model.trainTime,
              color: chartColors[index],
              isVerified: model.isVerified,
            }))}
          />

          <DotBars
            title="Complexity proxy"
            note="Storage and parameter counts remain provisional until the final export is wired in."
            unit="M"
            precision={2}
            data={modelResults.map((model, index) => ({
              label: model.label,
              meta: model.complexityLabel,
              value: model.complexity,
              color: chartColors[index],
              isVerified: model.isVerified,
            }))}
          />

          <DotPlot
            title="Performance vs training data"
            note="Dot-style placeholder curves; surface ready for real benchmark data."
            series={trainingDataTrend}
            xLabel="training data used (%)"
            yLabel="mRMSE"
          />
        </div>

        <div className="notes-column">
          {benchmarkNotes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </div>
      </GhostSection>

      <GhostSection
        id="quantum"
        index="04"
        eyebrow="Verified result"
        title="The clearest part of the story is the best hybrid checkpoint and how it behaves."
        summary="This section is grounded in saved repo artifacts: validation trajectory, checkpoint summary, and gas-level holdout error."
      >
        <div className="paper-grid paper-grid-2">
          <DotPlot
            title="Validation trajectory across epochs"
            note="Verified from the stored training history."
            series={validationCurve}
            xLabel="epoch"
            yLabel="validation mRMSE"
          />

          <article className="paper-stats">
            <div>
              <span>Best epoch</span>
              <strong>6</strong>
            </div>
            <div>
              <span>Best training-phase validation</span>
              <strong>0.2908</strong>
            </div>
            <div>
              <span>Re-evaluated validation</span>
              <strong>0.2936</strong>
            </div>
            <div>
              <span>Holdout mRMSE</span>
              <strong>0.2994</strong>
            </div>
          </article>
        </div>

        <div className="gas-strip">
          {perGasMetrics.map((metric) => (
            <article key={metric.gas} className="gas-strip-item">
              <p className="gas-strip-name">{metric.gas}</p>
              <p className="gas-strip-metrics">RMSE {metric.rmse.toFixed(3)} · MAE {metric.mae.toFixed(3)}</p>
            </article>
          ))}
        </div>
      </GhostSection>

      <GhostSection
        id="close"
        index="05"
        eyebrow="Close"
        title="This gives the judges a controlled scientific narrative with one verified quantum claim at the center."
        summary="The remaining work is mostly data replacement: swap the placeholder model-comparison values with your final exported benchmarks and the presentation is ready."
      >
        <div className="paper-grid paper-grid-2">
          <article className="paper-copy paper-copy-strong">
            <p className="paper-copy-label">What the website already does</p>
            <p>
              It frames the problem, the model family, the benchmark surfaces, and the saved quantum checkpoint in a
              style that matches the reference much more closely.
            </p>
          </article>

          <article className="paper-copy">
            <p className="paper-copy-label">What to replace next</p>
            <p>
              Feed real train-time, parameter, and sample-efficiency tables into the existing chart components and the
              site becomes fully presentation-ready without another structural redesign.
            </p>
          </article>
        </div>
      </GhostSection>
    </main>
  );
}
