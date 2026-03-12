import styles from "./page.module.css";
import { gases, modelRoster, performance, spectrumSeries, storyline } from "@/app/lib/project-data";

const valueBounds = spectrumSeries.reduce(
  (accumulator, point) => {
    return {
      min: Math.min(accumulator.min, point.observed, point.retrieved),
      max: Math.max(accumulator.max, point.observed, point.retrieved)
    };
  },
  { min: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY }
);

const chartWidth = 620;
const chartHeight = 260;
const chartPadding = 28;

function projectPoint(wavelength: number, value: number) {
  const minX = spectrumSeries[0]?.wavelength ?? 0.5;
  const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 5;
  const x = chartPadding + ((wavelength - minX) / (maxX - minX)) * (chartWidth - chartPadding * 2);
  const y =
    chartHeight -
    chartPadding -
    ((value - valueBounds.min) / (valueBounds.max - valueBounds.min)) * (chartHeight - chartPadding * 2);

  return `${x.toFixed(2)},${y.toFixed(2)}`;
}

const observedLine = spectrumSeries
  .map((point) => projectPoint(point.wavelength, point.observed))
  .join(" ");

const retrievedLine = spectrumSeries
  .map((point) => projectPoint(point.wavelength, point.retrieved))
  .join(" ");

const maxPerGasRmse = Math.max(...performance.perGasRmse.map((entry) => entry.value));

export default function VariantOnePage() {
  return (
    <main className={`${styles.page} page`}>
      <div className={`shell ${styles.shell}`}>
        <section className={styles.hero}>
          <article className={styles.masthead}>
            <div className={styles.docketRow}>
              <span className={styles.kicker}>Mission Ledger / official briefing deck</span>
              <span className={styles.stamp}>verified checkpoint</span>
            </div>
            <h1>ExoBiome</h1>
            <p className={styles.lead}>
              Quantifying five atmospheric gases from Ariel transmission spectroscopy and auxiliary
              planetary context, then framing their co-presence as a biosignature evidence problem
              rather than a single-gas headline.
            </p>
            <div className={styles.memo}>
              <div>
                <span>Core dataset</span>
                <strong>Ariel Data Challenge 2023</strong>
              </div>
              <div>
                <span>Hybrid model</span>
                <strong>Residual 1D encoder + 8-qubit correction branch</strong>
              </div>
              <div>
                <span>Demo mode</span>
                <strong>Readable in under 2 minutes</strong>
              </div>
            </div>
          </article>

          <aside className={styles.controlStack}>
            <section className={styles.metricCard}>
              <span className={styles.cardLabel}>Holdout mRMSE</span>
              <strong>{performance.holdout.toFixed(6)}</strong>
              <p>Re-evaluated from `best_model.pt` on {performance.rows.toLocaleString()} holdout rows.</p>
            </section>
            <section className={styles.quickFacts}>
              <div>
                <span>Training-phase best val</span>
                <strong>{performance.bestTrainingVal.toFixed(6)}</strong>
              </div>
              <div>
                <span>Validation after stop</span>
                <strong>{performance.validation.toFixed(6)}</strong>
              </div>
              <div>
                <span>Targets</span>
                <strong>H2O / CO2 / CO / CH4 / NH3</strong>
              </div>
            </section>
          </aside>
        </section>

        <section className={styles.overviewGrid}>
          <article className={styles.panel}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionIndex}>01</span>
              <h2>Biosignature ledger</h2>
            </div>
            <div className={styles.gasGrid}>
              {gases.map((gas) => (
                <article key={gas.key} className={styles.gasCard}>
                  <header>
                    <span className={styles.formula}>{gas.key}</span>
                    <strong>{gas.label}</strong>
                  </header>
                  <p>{gas.note}</p>
                  <small>{gas.role}</small>
                  <div className={styles.gasHighlight}>{gas.highlight}</div>
                </article>
              ))}
            </div>
            <p className={styles.footnote}>
              Scientific caveat: the page presents a five-gas context signal. It does not claim that
              any single abundance alone proves biology.
            </p>
          </article>

          <article className={styles.panel}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionIndex}>02</span>
              <h2>Model field lineup</h2>
            </div>
            <div className={styles.modelList}>
              {modelRoster.map((model, index) => (
                <article key={model.name} className={styles.modelCard}>
                  <div className={styles.modelMeta}>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <small>{model.position}</small>
                  </div>
                  <div>
                    <h3>{model.name}</h3>
                    <p>{model.summary}</p>
                  </div>
                  <strong>{model.className}</strong>
                </article>
              ))}
            </div>
          </article>
        </section>

        <section className={styles.analysisGrid}>
          <article className={`${styles.panel} ${styles.spectrumPanel}`}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionIndex}>03</span>
              <h2>Transmission briefing</h2>
            </div>
            <p className={styles.sectionIntro}>
              The route keeps the spectrum graphic restrained and legible: one observed curve, one
              retrieved curve, and just enough annotation to support the spoken pitch.
            </p>
            <div className={styles.chartFrame}>
              <svg
                viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                role="img"
                aria-label="Observed and retrieved transmission spectroscopy curves"
              >
                <rect x="0" y="0" width={chartWidth} height={chartHeight} rx="24" fill="rgba(253, 251, 245, 0.78)" />
                {[0, 1, 2, 3].map((step) => {
                  const y = chartPadding + step * ((chartHeight - chartPadding * 2) / 3);
                  return (
                    <line
                      key={step}
                      x1={chartPadding}
                      x2={chartWidth - chartPadding}
                      y1={y}
                      y2={y}
                      stroke="rgba(21, 47, 54, 0.12)"
                      strokeDasharray="5 6"
                    />
                  );
                })}
                <polyline
                  points={observedLine}
                  fill="none"
                  stroke="rgba(121, 71, 58, 0.72)"
                  strokeWidth="3.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <polyline
                  points={retrievedLine}
                  fill="none"
                  stroke="rgba(13, 95, 99, 1)"
                  strokeWidth="3.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <text x={chartPadding} y={18} fontSize="12" fill="rgba(21, 47, 54, 0.6)">
                  normalized transit depth
                </text>
                <text x={chartPadding} y={chartHeight - 8} fontSize="12" fill="rgba(21, 47, 54, 0.6)">
                  0.5 μm
                </text>
                <text x={chartWidth - chartPadding - 34} y={chartHeight - 8} fontSize="12" fill="rgba(21, 47, 54, 0.6)">
                  5.0 μm
                </text>
                <rect x="108" y="44" width="72" height="26" rx="13" fill="rgba(13, 95, 99, 0.1)" />
                <text x="122" y="61" fontSize="11" fill="rgba(13, 95, 99, 0.88)">
                  H2O band
                </text>
                <rect x="322" y="138" width="78" height="26" rx="13" fill="rgba(121, 71, 58, 0.1)" />
                <text x="338" y="155" fontSize="11" fill="rgba(121, 71, 58, 0.84)">
                  CO2 / CO
                </text>
              </svg>
            </div>
            <div className={styles.legend}>
              <span><i className={styles.observedSwatch} /> observed bins</span>
              <span><i className={styles.retrievedSwatch} /> retrieved profile</span>
            </div>
          </article>

          <article className={`${styles.panel} ${styles.pipelinePanel}`}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionIndex}>04</span>
              <h2>Architecture chain</h2>
            </div>
            <div className={styles.timeline}>
              {storyline.map((step, index) => (
                <article key={step.title} className={styles.timelineCard}>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <div>
                    <h3>{step.title}</h3>
                    <p>{step.detail}</p>
                  </div>
                </article>
              ))}
            </div>
            <div className={styles.quantumCallout}>
              <div>
                <span>Quantum branch</span>
                <strong>8 qubits, gated residual correction</strong>
              </div>
              <p>
                The classical head remains the stable backbone. The quantum block contributes a
                per-target corrective term instead of acting as the sole bottleneck.
              </p>
            </div>
          </article>
        </section>

        <section className={styles.verificationGrid}>
          <article className={styles.panel}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionIndex}>05</span>
              <h2>Verified checkpoint metrics</h2>
            </div>
            <div className={styles.metricLedger}>
              <div>
                <span>Training best</span>
                <strong>{performance.bestTrainingVal.toFixed(6)}</strong>
                <small>validation mRMSE during epoch {performance.bestEpoch}</small>
              </div>
              <div>
                <span>Validation</span>
                <strong>{performance.validation.toFixed(6)}</strong>
                <small>post-stop re-evaluation</small>
              </div>
              <div>
                <span>Holdout</span>
                <strong>{performance.holdout.toFixed(6)}</strong>
                <small>best verified external summary</small>
              </div>
            </div>
          </article>

          <article className={styles.panel}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionIndex}>06</span>
              <h2>Per-gas error profile</h2>
            </div>
            <div className={styles.barList}>
              {performance.perGasRmse.map((entry) => (
                <div key={entry.gas} className={styles.barRow}>
                  <div className={styles.barHeader}>
                    <span>{entry.gas}</span>
                    <strong>{entry.value.toFixed(3)}</strong>
                  </div>
                  <div className={styles.barTrack}>
                    <div
                      className={styles.barFill}
                      style={{ width: `${(entry.value / maxPerGasRmse) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </article>
        </section>
      </div>
    </main>
  );
}
