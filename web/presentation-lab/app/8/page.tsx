import {
  gases,
  modelRoster,
  performance,
  projectFacts,
  spectrumSeries,
  storyline
} from "@/app/lib/project-data";
import styles from "./page.module.css";

function buildPath(field: "observed" | "retrieved", width: number, height: number, padding: number) {
  const minX = spectrumSeries[0]?.wavelength ?? 0;
  const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 1;
  const values = spectrumSeries.flatMap((entry) => [entry.observed, entry.retrieved]);
  const minY = Math.min(...values) - 0.015;
  const maxY = Math.max(...values) + 0.015;

  return spectrumSeries
    .map((entry, index) => {
      const x = padding + ((entry.wavelength - minX) / (maxX - minX)) * (width - padding * 2);
      const y = height - padding - ((entry[field] - minY) / (maxY - minY)) * (height - padding * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

const spectrumWidth = 620;
const spectrumHeight = 320;
const spectrumPadding = 34;
const observedPath = buildPath("observed", spectrumWidth, spectrumHeight, spectrumPadding);
const retrievedPath = buildPath("retrieved", spectrumWidth, spectrumHeight, spectrumPadding);
const maxPerGasRmse = Math.max(...performance.perGasRmse.map((entry) => entry.value));

const guideSteps = [
  "Open with the problem: one gas is a hint, five gases together are a stronger biosignature context signal.",
  "Show the dataset and pipeline: Ariel transmission spectra, 52 bins, 4 channels, 8 auxiliary features.",
  "Land the result fast: 0.2908 best training validation and 0.2994 verified holdout mRMSE.",
  "Close on novelty: the classical encoder stays in charge while the 8-qubit branch adds a gated residual correction."
];

export default function VariantEightPage() {
  return (
    <main className={styles.page}>
      <div className={styles.atmosphere} aria-hidden="true" />
      <div className={styles.grain} aria-hidden="true" />

      <section className={styles.hero}>
        <div className={styles.heroText}>
          <span className={styles.kicker}>Exhibit 08 · Science-center installation for HACK-4-SAGES 2026</span>
          <h1>The Biosignature Hall</h1>
          <p className={styles.lead}>
            An immersive route for presenting <strong>ExoBiome</strong> as a scientific artifact: Ariel
            transmission spectra enter the gallery, a hybrid quantum regressor interprets them, and five
            atmospheric molecules leave as a disciplined biosignature story for judges who care about rigor.
          </p>

          <div className={styles.factRibbon}>
            {projectFacts.map((fact) => (
              <div key={fact.label} className={styles.factTile}>
                <span>{fact.label}</span>
                <strong>{fact.value}</strong>
              </div>
            ))}
          </div>
        </div>

        <aside className={styles.heroPanel}>
          <div className={styles.heroPanelInner}>
            <span className={styles.panelLabel}>Curator placard</span>
            <h2>Why this route works for a two-minute demo</h2>
            <p>
              It feels memorable and atmospheric like a museum installation, but every large gesture is
              tied to a concrete scientific fact already verified in the repository.
            </p>

            <div className={styles.metricStack}>
              <article className={styles.metricCard}>
                <span>Training val mRMSE</span>
                <strong>{performance.bestTrainingVal.toFixed(4)}</strong>
                <small>best recorded during stage-two training</small>
              </article>
              <article className={styles.metricCard}>
                <span>Validation mRMSE</span>
                <strong>{performance.validation.toFixed(4)}</strong>
                <small>re-evaluated checkpoint metric</small>
              </article>
              <article className={styles.metricCard}>
                <span>Holdout mRMSE</span>
                <strong>{performance.holdout.toFixed(4)}</strong>
                <small>{performance.rows.toLocaleString()} rows from `best_model.pt`</small>
              </article>
            </div>
          </div>
        </aside>
      </section>

      <section className={styles.galleryRow}>
        <article className={styles.vitrine}>
          <div className={styles.vitrineHead}>
            <span className={styles.panelLabel}>Gallery centerpiece</span>
            <h2>Five molecules under glass</h2>
            <p>
              The exhibit never oversells a single gas. It frames H2O, CO2, CO, CH4, and NH3 as a joint
              atmospheric context that becomes meaningful in combination.
            </p>
          </div>

          <div className={styles.gasConstellation}>
            <div className={styles.planetHalo} aria-hidden="true">
              <div className={styles.coreGlow} />
              <div className={styles.orbitRingOne} />
              <div className={styles.orbitRingTwo} />
            </div>

            <div className={styles.gasGrid}>
              {gases.map((gas) => (
                <article key={gas.key} className={styles.gasCard}>
                  <span className={styles.gasFormula}>{gas.key}</span>
                  <strong>{gas.label}</strong>
                  <p>{gas.note}</p>
                  <small>{gas.highlight}</small>
                </article>
              ))}
            </div>
          </div>
        </article>

        <aside className={styles.sideColumn}>
          <article className={styles.plaque}>
            <span className={styles.panelLabel}>Core dataset</span>
            <h3>Ariel Data Challenge 2023</h3>
            <p>
              The majority of the project converged on the Ariel challenge data, so this exhibit treats it
              as the single authoritative demonstration substrate.
            </p>
          </article>

          <article className={styles.plaque}>
            <span className={styles.panelLabel}>Model family line-up</span>
            <div className={styles.modelList}>
              {modelRoster.map((model) => (
                <div key={model.name} className={styles.modelItem}>
                  <strong>{model.name}</strong>
                  <span>
                    {model.className} · {model.position}
                  </span>
                  <p>{model.summary}</p>
                </div>
              ))}
            </div>
          </article>
        </aside>
      </section>

      <section className={styles.galleryRow}>
        <article className={styles.spectrumRoom}>
          <div className={styles.roomHeader}>
            <span className={styles.panelLabel}>Observation chamber</span>
            <h2>From transmission trace to retrieved abundances</h2>
            <p>
              The spectral display is intentionally positioned like a museum vitrine: a reconstructed
              measurement on top, the model&apos;s retrieval beneath it, and enough captioning to satisfy
              scientific judges without slowing the live talk.
            </p>
          </div>

          <div className={styles.spectrumFrame}>
            <svg viewBox={`0 0 ${spectrumWidth} ${spectrumHeight}`} role="img" aria-label="Transmission spectrum exhibit">
              <rect x="0" y="0" width={spectrumWidth} height={spectrumHeight} rx="28" className={styles.spectrumBackdrop} />
              {Array.from({ length: 5 }, (_, index) => {
                const y = spectrumPadding + index * ((spectrumHeight - spectrumPadding * 2) / 4);
                return (
                  <line
                    key={index}
                    x1={spectrumPadding}
                    x2={spectrumWidth - spectrumPadding}
                    y1={y}
                    y2={y}
                    className={styles.gridLine}
                  />
                );
              })}
              <path d={observedPath} className={styles.observedLine} />
              <path d={retrievedPath} className={styles.retrievedLine} />
              <text x={spectrumPadding} y="24" className={styles.axisText}>
                normalized transit depth
              </text>
              <text x={spectrumPadding} y={spectrumHeight - 10} className={styles.axisText}>
                0.5 μm
              </text>
              <text x={spectrumWidth - spectrumPadding - 34} y={spectrumHeight - 10} className={styles.axisText}>
                5.0 μm
              </text>
            </svg>
          </div>
        </article>

        <aside className={styles.rmsePanel}>
          <span className={styles.panelLabel}>Specimen labels</span>
          <h3>Per-gas holdout RMSE</h3>
          <div className={styles.rmseList}>
            {performance.perGasRmse.map((entry) => (
              <div key={entry.gas} className={styles.rmseRow}>
                <div className={styles.rmseMeta}>
                  <strong>{entry.gas}</strong>
                  <span>{entry.value.toFixed(3)}</span>
                </div>
                <div className={styles.rmseTrack}>
                  <div
                    className={styles.rmseFill}
                    style={{ width: `${(entry.value / maxPerGasRmse) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </aside>
      </section>

      <section className={styles.architectureWing}>
        <div className={styles.wingHeading}>
          <span className={styles.panelLabel}>Behind the glass</span>
          <h2>Architecture corridor</h2>
          <p>
            The exhibit walks visitors through the model as if they were moving room to room: the
            classical retrieval backbone remains dominant, while the quantum branch acts as a carefully
            gated residual correction rather than a gimmicky bottleneck.
          </p>
        </div>

        <div className={styles.storyGrid}>
          {storyline.map((step, index) => (
            <article key={step.title} className={styles.storyCard}>
              <span className={styles.storyIndex}>0{index + 1}</span>
              <strong>{step.title}</strong>
              <p>{step.detail}</p>
            </article>
          ))}
        </div>

        <div className={styles.architecturePlaque}>
          <div>
            <span className={styles.panelLabel}>Interpretation</span>
            <h3>Residual 1D encoder + 8-qubit quantum correction branch</h3>
          </div>
          <p>
            The presentation language stays plain: residual convolutional encoder, attention pooling,
            classical regression head, then an 8-qubit gated correction branch that fine-tunes the
            five-gas output vector instead of replacing the classical signal path.
          </p>
        </div>
      </section>

      <section className={styles.guideSection}>
        <div className={styles.guideHeader}>
          <span className={styles.panelLabel}>Curator script</span>
          <h2>A clean two-minute walkthrough</h2>
        </div>
        <div className={styles.guideGrid}>
          {guideSteps.map((step, index) => (
            <article key={step} className={styles.guideCard}>
              <span>Step {index + 1}</span>
              <p>{step}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
