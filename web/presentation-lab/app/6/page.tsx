import { gases, modelRoster, performance, projectFacts, spectrumSeries, storyline } from "@/app/lib/project-data";
import styles from "./page.module.css";

function buildPath(field: "observed" | "retrieved") {
  const width = 1400;
  const height = 300;
  const padX = 24;
  const padY = 30;
  const minX = spectrumSeries[0]?.wavelength ?? 0.5;
  const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 5.0;
  const values = spectrumSeries.flatMap((entry) => [entry.observed, entry.retrieved]);
  const minY = Math.min(...values) - 0.02;
  const maxY = Math.max(...values) + 0.02;

  return spectrumSeries
    .map((entry) => {
      const x = padX + ((entry.wavelength - minX) / (maxX - minX)) * (width - padX * 2);
      const y = height - padY - ((entry[field] - minY) / (maxY - minY)) * (height - padY * 2);
      return `${entry === spectrumSeries[0] ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

const observedPath = buildPath("observed");
const retrievedPath = buildPath("retrieved");

const metrics = [
  {
    label: "training-phase validation",
    value: performance.bestTrainingVal.toFixed(6),
    note: `best epoch ${performance.bestEpoch}`
  },
  {
    label: "re-evaluated validation",
    value: performance.validation.toFixed(6),
    note: "from best_model.pt"
  },
  {
    label: "re-evaluated holdout",
    value: performance.holdout.toFixed(6),
    note: `${performance.rows.toLocaleString()} rows`
  }
];

const stageCues = [
  "Lead with the scientific framing: joint abundance estimation, not one attractive molecule.",
  "Land the one number that matters: 0.299376 holdout mRMSE on Ariel ADC2023.",
  "Explain the hybrid: residual 1D encoder first, 8-qubit correction branch second."
];

export default function VariantSixPage() {
  return (
    <main className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.frame}>
          <p className={styles.eyebrow}>Monolith Keynote / Route 06 / Two-minute live deck</p>
          <div className={styles.heroGrid}>
            <div>
              <p className={styles.leadTag}>Ariel Data Challenge 2023</p>
              <h1>
                Quantify
                <br />
                five gases
                <br />
                before you
                <br />
                say biosignature.
              </h1>
            </div>
            <div className={styles.heroMetric}>
              <span>verified holdout mRMSE</span>
              <strong>0.299376</strong>
              <p>
                Best confirmed checkpoint in this repo. Residual 1D encoder plus an 8-qubit quantum
                correction branch.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className={styles.band}>
        <div className={styles.frame}>
          <div className={styles.bandIndex}>01</div>
          <div className={styles.bandBody}>
            <h2>The question is atmospheric context, not a single headline molecule.</h2>
            <p>
              ExoBiome estimates H2O, CO2, CO, CH4, and NH3 together from transmission spectroscopy
              plus auxiliary planetary context, then treats co-presence as a biosignature evidence
              problem rather than proof of life.
            </p>
            <div className={styles.factRun}>
              {projectFacts.map((fact) => (
                <span key={fact.label}>
                  <strong>{fact.label}</strong> {fact.value}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className={styles.gases}>
        <div className={styles.frame}>
          <p className={styles.sectionLabel}>02 / target vector</p>
          <div className={styles.gasStack}>
            {gases.map((gas) => (
              <article key={gas.key} className={styles.gasRow}>
                <strong>{gas.key}</strong>
                <div>
                  <span>{gas.label}</span>
                  <p>{gas.highlight}</p>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.spectrumBand}>
        <div className={styles.frame}>
          <div className={styles.spectrumHeader}>
            <p className={styles.sectionLabel}>03 / one visual argument</p>
            <h2>From one spectrum, to one five-gas vector.</h2>
          </div>
          <svg viewBox="0 0 1400 340" className={styles.spectrum} role="img" aria-label="Observed versus retrieved transmission spectrum">
            {[0, 1, 2, 3].map((line) => (
              <line
                key={line}
                x1="0"
                x2="1400"
                y1={42 + line * 72}
                y2={42 + line * 72}
                className={styles.gridLine}
              />
            ))}
            <path d={observedPath} className={styles.observed} />
            <path d={retrievedPath} className={styles.retrieved} />
            <text x="24" y="22">
              observed
            </text>
            <text x="124" y="22">
              retrieved
            </text>
            <text x="24" y="326">
              0.5 μm
            </text>
            <text x="1290" y="326">
              5.0 μm
            </text>
          </svg>
        </div>
      </section>

      <section className={styles.metricBand}>
        <div className={styles.frame}>
          <p className={styles.sectionLabel}>04 / verified checkpoints</p>
          <div className={styles.metricStrip}>
            {metrics.map((metric) => (
              <article key={metric.label} className={styles.metricColumn}>
                <span>{metric.label}</span>
                <strong>{metric.value}</strong>
                <p>{metric.note}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.lineupBand}>
        <div className={styles.frame}>
          <p className={styles.sectionLabel}>05 / model lineup</p>
          <div className={styles.lineupList}>
            {modelRoster.map((model, index) => (
              <article key={model.name} className={styles.lineupItem}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{model.name}</strong>
                <em>{model.className}</em>
                <p>{model.summary}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.architectureBand}>
        <div className={styles.frame}>
          <div className={styles.architectureHeader}>
            <p className={styles.sectionLabel}>06 / hybrid architecture</p>
            <h2>Residual 1D encoder first. 8-qubit correction branch second.</h2>
          </div>
          <div className={styles.storyline}>
            {storyline.map((step, index) => (
              <article key={step.title} className={styles.storyStep}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <div>
                  <strong>{step.title}</strong>
                  <p>{step.detail}</p>
                </div>
              </article>
            ))}
          </div>
          <div className={styles.wireField} aria-hidden="true">
            {Array.from({ length: 8 }, (_, index) => (
              <span key={index} />
            ))}
          </div>
        </div>
      </section>

      <section className={styles.close}>
        <div className={styles.frame}>
          <p className={styles.sectionLabel}>07 / delivery</p>
          <h2>Say less. Show the number. Show the structure. Stop.</h2>
          <ol className={styles.cueList}>
            {stageCues.map((cue) => (
              <li key={cue}>{cue}</li>
            ))}
          </ol>
          <p className={styles.closeNote}>
            This route is deliberately anti-dashboard: giant horizontal bands, one claim per panel,
            and enough typographic force to work as a live spoken demo rather than a website.
          </p>
        </div>
      </section>
    </main>
  );
}
