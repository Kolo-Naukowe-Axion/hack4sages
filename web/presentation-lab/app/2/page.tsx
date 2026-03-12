import styles from "./page.module.css";
import { gases, modelRoster, performance, projectFacts, spectrumSeries, storyline } from "@/app/lib/project-data";

const chartWidth = 620;
const chartHeight = 250;
const chartPadding = 20;

const values = spectrumSeries.flatMap((sample) => [sample.observed, sample.retrieved]);
const minY = Math.min(...values) - 0.02;
const maxY = Math.max(...values) + 0.02;

function projectPoint(wavelength: number, value: number) {
  const minX = spectrumSeries[0]?.wavelength ?? 0;
  const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 1;
  const x = chartPadding + ((wavelength - minX) / (maxX - minX)) * (chartWidth - chartPadding * 2);
  const y =
    chartHeight - chartPadding - ((value - minY) / (maxY - minY)) * (chartHeight - chartPadding * 2);
  return `${x.toFixed(2)},${y.toFixed(2)}`;
}

const observedPath = spectrumSeries.map((sample) => projectPoint(sample.wavelength, sample.observed)).join(" ");
const retrievedPath = spectrumSeries.map((sample) => projectPoint(sample.wavelength, sample.retrieved)).join(" ");
const maxRmse = Math.max(...performance.perGasRmse.map((entry) => entry.value));

export default function VariantTwoPage() {
  return (
    <main className={styles.page}>
      <div className={styles.gridAura} />
      <div className={styles.shell}>
        <section className={styles.hero}>
          <div className={styles.heroCopy}>
            <p className={styles.kicker}>Orbital Console / judge-ready telemetry deck</p>
            <h1>Five-gas biosignature retrieval, framed as a live mission console.</h1>
            <p className={styles.lead}>
              Ariel Data Challenge 2023 transmission spectra are fused with auxiliary planetary context,
              then routed through a residual 1D encoder and an 8-qubit quantum correction branch. The
              result is a fast, credible two-minute story for scientist judges: what went in, what was
              learned, and why the quantum model matters.
            </p>
            <div className={styles.callouts}>
              {projectFacts.map((fact) => (
                <div key={fact.label} className={styles.callout}>
                  <span>{fact.label}</span>
                  <strong>{fact.value}</strong>
                </div>
              ))}
            </div>
          </div>
          <div className={styles.commandPanel}>
            <div className={styles.panelHeader}>
              <span className={styles.panelTag}>Live metrics</span>
              <span className={styles.status}>Synced</span>
            </div>
            <div className={styles.metricStack}>
              <div className={styles.metricCard}>
                <span>Training best validation</span>
                <strong>{performance.bestTrainingVal.toFixed(6)}</strong>
                <small>epoch {performance.bestEpoch}</small>
              </div>
              <div className={styles.metricCard}>
                <span>Re-evaluated validation</span>
                <strong>{performance.validation.toFixed(6)}</strong>
                <small>best checkpoint replay</small>
              </div>
              <div className={styles.metricCard}>
                <span>Holdout mRMSE</span>
                <strong>{performance.holdout.toFixed(6)}</strong>
                <small>{performance.rows.toLocaleString()} rows</small>
              </div>
            </div>
            <div className={styles.commandFooter}>
              <span>Objective</span>
              <p>
                Quantify <strong>H2O</strong>, <strong>CO2</strong>, <strong>CO</strong>,{" "}
                <strong>CH4</strong>, and <strong>NH3</strong> jointly, treating co-presence as
                biosignature evidence rather than standalone proof.
              </p>
            </div>
          </div>
        </section>

        <section className={styles.telemetryRow}>
          <article className={styles.spectrumCard}>
            <div className={styles.cardHeading}>
              <div>
                <p className={styles.panelTag}>Spectral telemetry</p>
                <h2>Transmission fit overlay</h2>
              </div>
              <p className={styles.caption}>Observed trace vs retrieved structure across 0.5-5.0 um.</p>
            </div>
            <svg
              className={styles.chart}
              viewBox={`0 0 ${chartWidth} ${chartHeight}`}
              role="img"
              aria-label="Transmission spectrum overlay"
            >
              <rect width={chartWidth} height={chartHeight} rx="22" fill="rgba(7, 14, 28, 0.92)" />
              {[0, 1, 2, 3].map((step) => {
                const y = chartPadding + step * ((chartHeight - chartPadding * 2) / 3);
                return (
                  <line
                    key={step}
                    x1={chartPadding}
                    x2={chartWidth - chartPadding}
                    y1={y}
                    y2={y}
                    stroke="rgba(145, 169, 212, 0.14)"
                    strokeDasharray="4 8"
                  />
                );
              })}
              {[1.4, 2.7, 4.3].map((marker) => {
                const x = Number(projectPoint(marker, maxY).split(",")[0]);
                return (
                  <line
                    key={marker}
                    x1={x}
                    x2={x}
                    y1={chartPadding}
                    y2={chartHeight - chartPadding}
                    stroke="rgba(77, 212, 199, 0.18)"
                    strokeDasharray="8 12"
                  />
                );
              })}
              <polyline
                points={observedPath}
                fill="none"
                stroke="#7ce4df"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points={retrievedPath}
                fill="none"
                stroke="#ffc95c"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <text x="24" y="24" fill="rgba(204, 215, 242, 0.72)" fontSize="12">
                normalized transit depth
              </text>
              <text x="24" y="236" fill="rgba(204, 215, 242, 0.56)" fontSize="12">
                0.5 um
              </text>
              <text x="548" y="236" fill="rgba(204, 215, 242, 0.56)" fontSize="12">
                5.0 um
              </text>
            </svg>
          </article>

          <aside className={styles.sideTelemetry}>
            <div className={styles.panelHeader}>
              <span className={styles.panelTag}>Gas channels</span>
              <span className={styles.caption}>Joint biosignature context</span>
            </div>
            <div className={styles.gasList}>
              {gases.map((gas, index) => (
                <div key={gas.key} className={styles.gasCard}>
                  <div className={styles.gasHead}>
                    <strong>{gas.key}</strong>
                    <span>0{index + 1}</span>
                  </div>
                  <p>{gas.role}</p>
                  <small>{gas.highlight}</small>
                </div>
              ))}
            </div>
          </aside>
        </section>

        <section className={styles.lowerGrid}>
          <article className={styles.modelsCard}>
            <div className={styles.cardHeading}>
              <div>
                <p className={styles.panelTag}>Model lineup</p>
                <h2>Benchmark stack used in the demo</h2>
              </div>
              <p className={styles.caption}>Classical baselines, challenge winner, and the custom quantum route.</p>
            </div>
            <div className={styles.modelList}>
              {modelRoster.map((model, index) => (
                <div key={model.name} className={styles.modelCard}>
                  <div className={styles.modelIndex}>0{index + 1}</div>
                  <div>
                    <strong>{model.name}</strong>
                    <span>{model.className}</span>
                    <p>{model.summary}</p>
                  </div>
                  <small>{model.position}</small>
                </div>
              ))}
            </div>
          </article>

          <article className={styles.architectureCard}>
            <div className={styles.cardHeading}>
              <div>
                <p className={styles.panelTag}>Architecture corridor</p>
                <h2>Retrieval pipeline in four stations</h2>
              </div>
              <p className={styles.caption}>Every stage optimized for demo readability.</p>
            </div>
            <div className={styles.storySteps}>
              {storyline.map((step, index) => (
                <div key={step.title} className={styles.storyStep}>
                  <div className={styles.storyIndex}>0{index + 1}</div>
                  <div>
                    <strong>{step.title}</strong>
                    <p>{step.detail}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className={styles.quantumBadge}>
              <span>Quantum branch</span>
              <strong>8-qubit residual correction</strong>
              <p>Per-target gating lets each gas receive a different quantum adjustment strength.</p>
            </div>
          </article>
        </section>

        <section className={styles.footerDeck}>
          <article className={styles.rmseCard}>
            <div className={styles.cardHeading}>
              <div>
                <p className={styles.panelTag}>Error distribution</p>
                <h2>Per-gas holdout RMSE</h2>
              </div>
              <p className={styles.caption}>Error profile of the verified best checkpoint.</p>
            </div>
            <div className={styles.rmseList}>
              {performance.perGasRmse.map((entry) => (
                <div key={entry.gas} className={styles.rmseRow}>
                  <div className={styles.rmseMeta}>
                    <strong>{entry.gas}</strong>
                    <span>{entry.value.toFixed(3)}</span>
                  </div>
                  <div className={styles.rmseTrack}>
                    <div className={styles.rmseFill} style={{ width: `${(entry.value / maxRmse) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </article>

          <article className={styles.demoCard}>
            <p className={styles.panelTag}>Two-minute demo cue sheet</p>
            <ol className={styles.demoList}>
              <li>Open on the live metrics block and say the repo-verified holdout result out loud: 0.299376.</li>
              <li>Point to the spectrum overlay and explain that the model consumes transmission spectroscopy plus 8 auxiliary features.</li>
              <li>Sweep through the five gases and make the scientific caution explicit: combined evidence, not proof of life.</li>
              <li>Close on the model lineup and the 8-qubit residual story as your differentiator for the hackathon.</li>
            </ol>
          </article>
        </section>
      </div>
    </main>
  );
}
