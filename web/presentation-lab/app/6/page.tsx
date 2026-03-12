import {
  gases,
  modelRoster,
  performance,
  projectFacts,
  routeVariants,
  spectrumSeries,
  storyline
} from "@/app/lib/project-data";
import styles from "./page.module.css";

function buildSpectrumPath(values: number[], width: number, height: number) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const xStep = width / Math.max(values.length - 1, 1);

  return values
    .map((value, index) => {
      const x = index * xStep;
      const y = height - ((value - min) / (max - min || 1)) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

const observedPath = buildSpectrumPath(
  spectrumSeries.map((point) => point.observed),
  760,
  240
);

const retrievedPath = buildSpectrumPath(
  spectrumSeries.map((point) => point.retrieved),
  760,
  240
);

const metricCards = [
  {
    label: "Training best",
    value: performance.bestTrainingVal.toFixed(6),
    note: "validation mRMSE during training"
  },
  {
    label: "Validation",
    value: performance.validation.toFixed(6),
    note: "re-evaluated from best_model.pt"
  },
  {
    label: "Holdout",
    value: performance.holdout.toFixed(6),
    note: `${performance.rows.toLocaleString()} rows, verified snapshot`
  }
];

const stageBeats = [
  "Open with the scientific question: can we quantify five gases jointly instead of showcasing one attractive peak.",
  "Show the one number the judges will remember: 0.299376 holdout mRMSE on Ariel ADC2023.",
  "Explain why the model matters: classical residual encoder, then an 8-qubit correction branch that refines the abundance vector."
];

const variant = routeVariants.find((item) => item.slug === "6");

export default function VariantSixPage() {
  return (
    <main className={`page ${styles.stagePage}`}>
      <div className="shell">
        <section className={styles.hero}>
          <div className={styles.heroBackdrop} aria-hidden="true" />
          <div className={styles.heroLead}>
            <span className={styles.kicker}>{variant?.framing ?? "two-minute keynote"}</span>
            <p className={styles.contextLine}>AXION • HACK-4-SAGES 2026 • ETH Zurich orbit</p>
            <h1>
              Five gases.
              <br />
              One spectrum.
              <br />
              <span>0.299376 holdout mRMSE.</span>
            </h1>
            <p className={styles.summary}>
              We frame biosignatures as a joint abundance problem on Ariel transmission spectroscopy:
              H2O, CO2, CO, CH4, and NH3 inferred together from the challenge dataset, auxiliary
              planetary features, and a hybrid quantum residual model.
            </p>

            <div className={styles.gasRail}>
              {gases.map((gas) => (
                <article key={gas.key} className={styles.gasCard}>
                  <strong>{gas.key}</strong>
                  <span>{gas.label}</span>
                </article>
              ))}
            </div>
          </div>

          <aside className={styles.heroBoard}>
            <div className={styles.boardTop}>
              <span>demo cue</span>
              <strong>judge-safe storyline</strong>
            </div>
            <ol className={styles.beatList}>
              {stageBeats.map((beat) => (
                <li key={beat}>{beat}</li>
              ))}
            </ol>
            <div className={styles.factStrip}>
              {projectFacts.map((fact) => (
                <div key={fact.label} className={styles.factTile}>
                  <span>{fact.label}</span>
                  <strong>{fact.value}</strong>
                </div>
              ))}
            </div>
          </aside>
        </section>

        <section className={styles.metricsSection}>
          {metricCards.map((metric, index) => (
            <article key={metric.label} className={styles.metricCard}>
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
              <p>{metric.note}</p>
              <i aria-hidden="true">{String(index + 1).padStart(2, "0")}</i>
            </article>
          ))}
        </section>

        <section className={styles.storyGrid}>
          <article className={styles.spectrumPanel}>
            <div className={styles.panelHeader}>
              <span>why this works on stage</span>
              <strong>One visual argument</strong>
            </div>
            <div className={styles.chartWrap}>
              <svg viewBox="0 0 760 280" className={styles.chart} role="img" aria-label="Spectrum fit">
                {[0, 1, 2, 3].map((line) => (
                  <line
                    key={line}
                    x1="0"
                    x2="760"
                    y1={24 + line * 64}
                    y2={24 + line * 64}
                    className={styles.gridLine}
                  />
                ))}
                <path d={observedPath} className={styles.observed} transform="translate(0 20)" />
                <path d={retrievedPath} className={styles.retrieved} transform="translate(0 20)" />
                <text x="0" y="272">
                  0.5 μm
                </text>
                <text x="675" y="272">
                  5.0 μm
                </text>
                <text x="0" y="16">
                  transmission spectrum
                </text>
              </svg>
            </div>
            <div className={styles.panelNarrative}>
              <p>
                The demo does not try to teach exoplanet spectroscopy from scratch. It shows one
                representative transmission spectrum, then pivots immediately to the five-gas vector
                we estimate from it.
              </p>
              <p>
                Ariel Data Challenge 2023 remains the core benchmark, so the judges can map the story
                directly onto a known scientific task rather than a toy showcase.
              </p>
            </div>
          </article>

          <article className={styles.architecturePanel}>
            <div className={styles.panelHeader}>
              <span>architecture</span>
              <strong>Residual encoder + 8-qubit correction</strong>
            </div>
            <div className={styles.pipeline}>
              {storyline.map((step, index) => (
                <div key={step.title} className={styles.pipelineCard}>
                  <i>{String(index + 1).padStart(2, "0")}</i>
                  <h2>{step.title}</h2>
                  <p>{step.detail}</p>
                </div>
              ))}
            </div>
            <div className={styles.quantumRibbon}>
              <span className={styles.ribbonTag}>hybrid block</span>
              <div className={styles.wires} aria-hidden="true">
                {Array.from({ length: 8 }, (_, index) => (
                  <span key={index} />
                ))}
              </div>
              <p>
                The classical backbone carries the bulk of the retrieval signal. The quantum branch
                acts as a gated residual correction, allowing each gas target to receive a different
                quantum adjustment strength.
              </p>
            </div>
          </article>
        </section>

        <section className={styles.lineupSection}>
          <div className={styles.sectionLead}>
            <span>model lineup</span>
            <h2>Four ways to tell the judges we did not stop at one baseline.</h2>
          </div>
          <div className={styles.lineupGrid}>
            {modelRoster.map((model, index) => (
              <article
                key={model.name}
                className={`${styles.lineupCard} ${
                  model.name === "Hybrid Quantum Residual" ? styles.lineupCardAccent : ""
                }`}
              >
                <div className={styles.lineupIndex}>{String(index + 1).padStart(2, "0")}</div>
                <div>
                  <span>{model.className}</span>
                  <h3>{model.name}</h3>
                  <strong>{model.position}</strong>
                </div>
                <p>{model.summary}</p>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
