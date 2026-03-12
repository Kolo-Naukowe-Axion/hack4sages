import styles from "./page.module.css";
import { gases, modelRoster, performance, projectFacts, spectrumSeries, storyline } from "@/app/lib/project-data";

function buildSpectrumLine(field: "observed" | "retrieved") {
  const width = 620;
  const height = 260;
  const paddingX = 34;
  const paddingY = 28;
  const xMin = spectrumSeries[0]?.wavelength ?? 0.5;
  const xMax = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 5;
  const values = spectrumSeries.flatMap((entry) => [entry.observed, entry.retrieved]);
  const yMin = Math.min(...values) - 0.02;
  const yMax = Math.max(...values) + 0.02;

  return spectrumSeries
    .map((entry) => {
      const x = paddingX + ((entry.wavelength - xMin) / (xMax - xMin)) * (width - paddingX * 2);
      const y = height - paddingY - ((entry[field] - yMin) / (yMax - yMin)) * (height - paddingY * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export default function VariantThreePage() {
  const maxRmse = Math.max(...performance.perGasRmse.map((entry) => entry.value));
  const observedLine = buildSpectrumLine("observed");
  const retrievedLine = buildSpectrumLine("retrieved");

  return (
    <main className={styles.page}>
      <div className={styles.board}>
        <header className={styles.header}>
          <div className={styles.headerLead}>
            <span className={styles.kicker}>Poster Session / Variant 3 / ETH-style judge board</span>
            <h1>ExoBiome</h1>
            <p className={styles.subtitle}>
              Joint biosignature quantification from Ariel transmission spectroscopy, packaged as a
              web-native conference poster for a two-minute scientific demo.
            </p>
          </div>
          <div className={styles.headerStats}>
            <div className={styles.metricTile}>
              <span>Best holdout mRMSE</span>
              <strong>{performance.holdout.toFixed(6)}</strong>
              <small>Re-evaluated on {performance.rows.toLocaleString()} holdout planets</small>
            </div>
            <div className={styles.metricTile}>
              <span>Validation checkpoint</span>
              <strong>{performance.validation.toFixed(6)}</strong>
              <small>Best confirmed `best_model.pt` validation metric</small>
            </div>
          </div>
        </header>

        <section className={styles.abstractStrip}>
          <article className={styles.abstractCard}>
            <p className={styles.abstractLabel}>Abstract</p>
            <p>
              We treat biosignature detection as a joint abundance problem rather than a single-gas
              claim. Using Ariel Data Challenge 2023 transmission spectra plus auxiliary stellar and
              planetary features, our hybrid model predicts <strong>H2O, CO2, CO, CH4, and NH3</strong>{" "}
              in one pass, then frames their co-presence as biosignature evidence rather than proof of life.
            </p>
          </article>
          <article className={styles.noteCard}>
            <p className={styles.abstractLabel}>Poster framing</p>
            <p>
              The visual language is intentionally academic: modular panels, figure captions, and
              committee-grade readability, but delivered as a responsive web page instead of a PDF poster.
            </p>
          </article>
        </section>

        <div className={styles.posterGrid}>
          <section className={styles.column}>
            <article className={styles.panel}>
              <div className={styles.panelHeading}>
                <h2>Problem Framing</h2>
                <span>01</span>
              </div>
              <div className={styles.factGrid}>
                {projectFacts.map((fact) => (
                  <div key={fact.label} className={styles.fact}>
                    <small>{fact.label}</small>
                    <strong>{fact.value}</strong>
                  </div>
                ))}
              </div>
              <div className={styles.captionBlock}>
                <p className={styles.captionTitle}>Judge-facing claim</p>
                <p>
                  The project is not “life detection” theater. It is a scientifically legible
                  abundance estimator that makes disequilibrium reasoning faster to inspect in a hackathon setting.
                </p>
              </div>
            </article>

            <article className={styles.panel}>
              <div className={styles.panelHeading}>
                <h2>Target Biosignature Set</h2>
                <span>02</span>
              </div>
              <div className={styles.gasList}>
                {gases.map((gas) => (
                  <div key={gas.key} className={styles.gasCard}>
                    <div className={styles.gasTopline}>
                      <strong>{gas.key}</strong>
                      <small>{gas.label}</small>
                    </div>
                    <p>{gas.note}</p>
                    <span>{gas.highlight}</span>
                  </div>
                ))}
              </div>
              <p className={styles.figureCaption}>
                Figure A. Five-gas panel used for all variants of the showcase, emphasizing context over isolated detections.
              </p>
            </article>

            <article className={styles.panel}>
              <div className={styles.panelHeading}>
                <h2>Pipeline</h2>
                <span>03</span>
              </div>
              <div className={styles.pipeline}>
                {storyline.map((step, index) => (
                  <div key={step.title} className={styles.pipelineStep}>
                    <div className={styles.stepIndex}>0{index + 1}</div>
                    <div>
                      <h3>{step.title}</h3>
                      <p>{step.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
              <p className={styles.figureCaption}>
                Figure B. End-to-end story compressed to four judge-readable steps, matching the actual repo architecture notes.
              </p>
            </article>
          </section>

          <section className={styles.column}>
            <article className={`${styles.panel} ${styles.figurePanel}`}>
              <div className={styles.panelHeading}>
                <h2>Representative Transmission Spectrum</h2>
                <span>04</span>
              </div>
              <svg className={styles.spectrum} viewBox="0 0 620 260" role="img" aria-label="Spectrum figure">
                <rect x="0" y="0" width="620" height="260" rx="24" fill="#fffaf1" />
                {[0, 1, 2, 3].map((index) => {
                  const y = 28 + index * 68;
                  return (
                    <line
                      key={index}
                      x1="34"
                      x2="586"
                      y1={y}
                      y2={y}
                      stroke="rgba(26, 40, 58, 0.14)"
                      strokeDasharray="5 8"
                    />
                  );
                })}
                <polyline
                  points={observedLine}
                  fill="none"
                  stroke="#bc5e44"
                  strokeWidth="4"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                <polyline
                  points={retrievedLine}
                  fill="none"
                  stroke="#19586a"
                  strokeWidth="4"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                <text x="34" y="22" className={styles.svgLabel}>
                  normalized transit depth
                </text>
                <text x="34" y="248" className={styles.svgLabel}>
                  0.5 μm
                </text>
                <text x="548" y="248" className={styles.svgLabel}>
                  5.0 μm
                </text>
                <g transform="translate(416 24)">
                  <rect x="0" y="0" width="160" height="52" rx="14" fill="rgba(255,255,255,0.82)" />
                  <line x1="14" y1="18" x2="46" y2="18" stroke="#bc5e44" strokeWidth="4" />
                  <text x="56" y="22" className={styles.svgLegend}>
                    observed
                  </text>
                  <line x1="14" y1="35" x2="46" y2="35" stroke="#19586a" strokeWidth="4" />
                  <text x="56" y="39" className={styles.svgLegend}>
                    retrieved
                  </text>
                </g>
              </svg>
              <p className={styles.figureCaption}>
                Figure C. Poster-style spectrum figure with measured and recovered trajectories, aligned to the Ariel transmission workflow.
              </p>
            </article>

            <article className={styles.panel}>
              <div className={styles.panelHeading}>
                <h2>Architecture Story</h2>
                <span>05</span>
              </div>
              <div className={styles.architecture}>
                <div className={styles.archBlock}>
                  <small>Classical branch</small>
                  <strong>Residual 1D encoder</strong>
                  <p>Four spectral channels, attention pooling, auxiliary encoder, and a direct regression head.</p>
                </div>
                <div className={styles.archConnector}>+</div>
                <div className={styles.archBlock}>
                  <small>Quantum branch</small>
                  <strong>8-qubit correction</strong>
                  <p>Gated residual pathway that refines gas-wise outputs rather than replacing the classical backbone.</p>
                </div>
              </div>
              <div className={styles.callout}>
                <strong>Training schedule</strong>
                <p>Classical pretrain, then hybrid fine-tune with quantum warmup, ramp, and early stopping at epoch {performance.bestEpoch}.</p>
              </div>
            </article>

            <article className={styles.panel}>
              <div className={styles.panelHeading}>
                <h2>Benchmark Snapshot</h2>
                <span>06</span>
              </div>
              <div className={styles.resultsGrid}>
                <div className={styles.resultCard}>
                  <small>Training-phase best validation</small>
                  <strong>{performance.bestTrainingVal.toFixed(6)}</strong>
                </div>
                <div className={styles.resultCard}>
                  <small>Re-evaluated validation</small>
                  <strong>{performance.validation.toFixed(6)}</strong>
                </div>
                <div className={styles.resultCard}>
                  <small>Re-evaluated holdout</small>
                  <strong>{performance.holdout.toFixed(6)}</strong>
                </div>
              </div>
              <div className={styles.barGroup}>
                {performance.perGasRmse.map((item) => (
                  <div key={item.gas} className={styles.barRow}>
                    <div className={styles.barLabel}>
                      <span>{item.gas}</span>
                      <small>{item.value.toFixed(3)}</small>
                    </div>
                    <div className={styles.barTrack}>
                      <div className={styles.barFill} style={{ width: `${(item.value / maxRmse) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
              <p className={styles.figureCaption}>
                Figure D. Error is lowest for CO and CH4 in this checkpoint; H2O and NH3 remain the hardest retrieval axes.
              </p>
            </article>
          </section>
        </div>

        <section className={styles.footerStrip}>
          <article className={styles.panel}>
            <div className={styles.panelHeading}>
              <h2>Model Lineup</h2>
              <span>07</span>
            </div>
            <div className={styles.modelRow}>
              {modelRoster.map((model) => (
                <div key={model.name} className={styles.modelCard}>
                  <small>{model.className}</small>
                  <strong>{model.name}</strong>
                  <span>{model.position}</span>
                  <p>{model.summary}</p>
                </div>
              ))}
            </div>
          </article>

          <article className={styles.panel}>
            <div className={styles.panelHeading}>
              <h2>Two-Minute Demo Script</h2>
              <span>08</span>
            </div>
            <ol className={styles.demoList}>
              <li>Start with the five-gas poster wall to show the biosignature framing.</li>
              <li>Point to the spectrum figure and say the input is Ariel transmission data plus auxiliary context.</li>
              <li>Land on the hybrid architecture card and note the 8-qubit branch is a gated residual, not hype bait.</li>
              <li>Finish on the mRMSE trio and the lineup panel: classical baselines, winner-style reference, then your quantum model.</li>
            </ol>
          </article>
        </section>
      </div>
    </main>
  );
}
