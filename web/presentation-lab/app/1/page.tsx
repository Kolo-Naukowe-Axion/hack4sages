import styles from "./page.module.css";
import { gases, modelRoster, performance, projectFacts, spectrumSeries, storyline } from "@/app/lib/project-data";

const plotWidth = 760;
const plotHeight = 260;
const plotPaddingX = 26;
const plotPaddingY = 22;

const values = spectrumSeries.flatMap((point) => [point.observed, point.retrieved]);
const minValue = Math.min(...values) - 0.02;
const maxValue = Math.max(...values) + 0.02;

function projectPoint(wavelength: number, value: number) {
  const minX = spectrumSeries[0]?.wavelength ?? 0.5;
  const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 5;
  const x = plotPaddingX + ((wavelength - minX) / (maxX - minX)) * (plotWidth - plotPaddingX * 2);
  const y =
    plotHeight -
    plotPaddingY -
    ((value - minValue) / (maxValue - minValue)) * (plotHeight - plotPaddingY * 2);

  return `${x.toFixed(2)},${y.toFixed(2)}`;
}

const observedLine = spectrumSeries.map((point) => projectPoint(point.wavelength, point.observed)).join(" ");
const retrievedLine = spectrumSeries.map((point) => projectPoint(point.wavelength, point.retrieved)).join(" ");

const metricRows = [
  { label: "Training best", value: performance.bestTrainingVal.toFixed(6) },
  { label: "Validation", value: performance.validation.toFixed(6) },
  { label: "Holdout", value: performance.holdout.toFixed(6) }
];

export default function VariantOnePage() {
  return (
    <main className={styles.page}>
      <div className={styles.sheet}>
        <section className={styles.leadRow}>
          <div className={styles.headlineColumn}>
            <p className={styles.strap}>Swiss Control Sheet / Route 01</p>
            <h1>Quantifying a five-gas biosignature stack from Ariel spectra.</h1>
            <p className={styles.dek}>
              This route is built like a committee board rather than a landing page: one headline,
              one metric column, one process band, and one hard comparison table. The point is to
              let a judge scan the scientific argument in under two minutes without fighting the layout.
            </p>
            <div className={styles.factLine}>
              {projectFacts.map((fact) => (
                <span key={fact.label}>
                  <strong>{fact.label}</strong> {fact.value}
                </span>
              ))}
            </div>
          </div>

          <aside className={styles.metricColumn}>
            <div className={styles.metricBlock}>
              <span className={styles.label}>Verified checkpoint</span>
              <strong>{performance.holdout.toFixed(6)}</strong>
              <p>Holdout mRMSE from `best_model.pt` on {performance.rows.toLocaleString()} rows.</p>
            </div>
            <dl className={styles.metricList}>
              {metricRows.map((metric) => (
                <div key={metric.label}>
                  <dt>{metric.label}</dt>
                  <dd>{metric.value}</dd>
                </div>
              ))}
            </dl>
          </aside>
        </section>

        <section className={styles.processBand}>
          {storyline.map((step, index) => (
            <article key={step.title} className={styles.processStep}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{step.title}</strong>
              <p>{step.detail}</p>
            </article>
          ))}
        </section>

        <section className={styles.analysisRow}>
          <div className={styles.visualBlock}>
            <div className={styles.blockHeader}>
              <span className={styles.label}>Observed vs retrieved transmission structure</span>
              <strong>Ariel Data Challenge 2023</strong>
            </div>
            <svg viewBox={`0 0 ${plotWidth} ${plotHeight}`} className={styles.chart} role="img" aria-label="Transmission spectrum fit">
              <rect x="0" y="0" width={plotWidth} height={plotHeight} fill="#ffffff" />
              {[0, 1, 2, 3].map((step) => {
                const y = plotPaddingY + step * ((plotHeight - plotPaddingY * 2) / 3);
                return (
                  <line
                    key={step}
                    x1={plotPaddingX}
                    x2={plotWidth - plotPaddingX}
                    y1={y}
                    y2={y}
                    stroke="rgba(0, 0, 0, 0.14)"
                    strokeDasharray="4 6"
                  />
                );
              })}
              <polyline points={observedLine} fill="none" stroke="rgba(0, 0, 0, 0.88)" strokeWidth="3" />
              <polyline points={retrievedLine} fill="none" stroke="#ff4b2b" strokeWidth="3" />
              <text x={plotPaddingX} y="18">normalized transit depth</text>
              <text x={plotPaddingX} y={plotHeight - 7}>0.5 um</text>
              <text x={plotWidth - plotPaddingX - 30} y={plotHeight - 7}>5.0 um</text>
            </svg>
            <p className={styles.caption}>
              The route keeps a single figure and uses it as evidence, not decoration.
            </p>
          </div>

          <div className={styles.annotationColumn}>
            <div className={styles.annotationBlock}>
              <span className={styles.label}>Architecture note</span>
              <h2>Residual 1D encoder first. 8-qubit correction second.</h2>
              <p>
                The hybrid quantum residual is presented as a disciplined extension of a strong
                classical backbone, not as a theatrical replacement for it. The correction branch
                adjusts the final abundance vector per target.
              </p>
            </div>

            <div className={styles.gasLedger}>
              {gases.map((gas) => (
                <div key={gas.key} className={styles.gasRow}>
                  <strong>{gas.key}</strong>
                  <span>{gas.label}</span>
                  <p>{gas.highlight}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className={styles.tableSection}>
          <div className={styles.tableHeader}>
            <div>
              <span className={styles.label}>Model comparison</span>
              <h2>Four-model lineup on the same retrieval story</h2>
            </div>
            <p>
              Baseline CNN and RF anchor the classical references. The winner-style NSF establishes
              the challenge-grade benchmark. The hybrid quantum residual is the team claim.
            </p>
          </div>

          <table className={styles.comparisonTable}>
            <thead>
              <tr>
                <th>Model</th>
                <th>Family</th>
                <th>Role in pitch</th>
                <th>Summary</th>
              </tr>
            </thead>
            <tbody>
              {modelRoster.map((model) => (
                <tr key={model.name}>
                  <td>{model.name}</td>
                  <td>{model.className}</td>
                  <td>{model.position}</td>
                  <td>{model.summary}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className={styles.bottomNote}>
            <span className={styles.label}>Scientific framing</span>
            <p>
              H2O, CO2, CO, CH4, and NH3 are presented as a combined atmospheric context signal.
              The page is explicit that this is biosignature evidence, not proof of life.
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}
