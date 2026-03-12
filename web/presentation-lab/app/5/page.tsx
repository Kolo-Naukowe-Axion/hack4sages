import styles from "./page.module.css";
import { gases, modelRoster, performance, spectrumSeries, storyline } from "@/app/lib/project-data";

const architecturePillars = [
  "Residual 1D spectral encoder",
  "Attention pooling over the transmission signal",
  "Classical regression head for the baseline prediction",
  "8-qubit quantum correction branch with per-target gating",
  "Joint abundance output for H2O, CO2, CO, CH4, and NH3"
];

const spectralMarkers = [
  { label: "H2O", wavelength: 1.4 },
  { label: "CH4", wavelength: 2.7 },
  { label: "CO2", wavelength: 4.3 }
];

function buildPath(field: "observed" | "retrieved") {
  const width = 900;
  const height = 260;
  const paddingX = 36;
  const paddingY = 28;
  const minX = spectrumSeries[0]?.wavelength ?? 0.5;
  const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 5.0;
  const values = spectrumSeries.flatMap((point) => [point.observed, point.retrieved]);
  const minY = Math.min(...values) - 0.02;
  const maxY = Math.max(...values) + 0.02;

  return spectrumSeries
    .map((point) => {
      const x = paddingX + ((point.wavelength - minX) / (maxX - minX)) * (width - paddingX * 2);
      const y = height - paddingY - ((point[field] - minY) / (maxY - minY)) * (height - paddingY * 2);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

export default function VariantFivePage() {
  return (
    <main className={`page ${styles.page}`}>
      <div className={`shell ${styles.editorial}`}>
        <section className={styles.masthead}>
          <div className={styles.issueStrip}>
            <span>Axion research notes</span>
            <span>Hackathon field edition</span>
            <span>ETH Zurich orbit</span>
          </div>
          <div className={styles.heroGrid}>
            <article className={styles.heroCopy}>
              <p className={styles.overline}>Data editorial / biosignature feature</p>
              <h1>Turning Ariel spectra into a five-gas biosignature story.</h1>
              <p className={styles.lede}>
                In two minutes, this route frames the project as a scientific feature piece: the Ariel
                Data Challenge 2023 provides the transmission spectroscopy backbone, the hybrid model
                estimates abundances for H2O, CO2, CO, CH4, and NH3, and the result is a cleaner way
                to talk about biosignature evidence as a joint atmospheric pattern instead of a single
                dramatic gas.
              </p>
            </article>

            <aside className={styles.scoreCard}>
              <span className={styles.scoreLabel}>Verified holdout mRMSE</span>
              <strong className={styles.scoreValue}>{performance.holdout.toFixed(4)}</strong>
              <p>
                Best confirmed checkpoint in this repo, re-evaluated on {performance.rows.toLocaleString()} holdout
                planets from the same Ariel-derived pipeline.
              </p>
              <div className={styles.scoreMeta}>
                <div>
                  <span>Training val</span>
                  <strong>{performance.bestTrainingVal.toFixed(6)}</strong>
                </div>
                <div>
                  <span>Validation</span>
                  <strong>{performance.validation.toFixed(6)}</strong>
                </div>
              </div>
            </aside>
          </div>
        </section>

        <section className={styles.openingSpread}>
          <article className={styles.dropCapBlock}>
            <p>
              <span className={styles.dropCap}>T</span>he editorial angle here is clarity. The judges do
              not need a sprawling website; they need a coherent scientific argument. Start with the
              dataset, move through the model family, land on the hybrid quantum residual, and end on
              the fact that biosignature reasoning is stronger when the atmosphere is discussed as a
              system. This design is built to let someone speak over the page without fighting it.
            </p>
          </article>
          <aside className={styles.sidebarNote}>
            <span className={styles.kicker}>What matters</span>
            <ul>
              <li>ADC2023 is the core experimental benchmark.</li>
              <li>The model predicts five abundances, not a binary label.</li>
              <li>The quantum path is a correction branch, not a gimmick bottleneck.</li>
            </ul>
          </aside>
        </section>

        <section className={styles.chapterBand}>
          {storyline.map((step, index) => (
            <article key={step.title} className={styles.chapterCard}>
              <span className={styles.chapterNumber}>0{index + 1}</span>
              <h2>{step.title}</h2>
              <p>{step.detail}</p>
            </article>
          ))}
        </section>

        <section className={styles.middleGrid}>
          <article className={styles.figureFeature}>
            <div className={styles.figureHeading}>
              <div>
                <span className={styles.kicker}>Feature figure</span>
                <h2>Observed transmission structure versus retrieved trend</h2>
              </div>
              <p>
                The purpose of the model is not to display a flashy chart. It is to convert a physically
                constrained transmission spectrum into five abundance estimates with enough stability to
                compare model families and support a biosignature narrative.
              </p>
            </div>

            <div className={styles.figurePanel}>
              <svg viewBox="0 0 900 260" role="img" aria-label="Transmission spectrum feature">
                <rect x="0" y="0" width="900" height="260" rx="26" fill="rgba(252,249,242,0.84)" />
                {[0, 1, 2, 3].map((step) => {
                  const y = 28 + step * 68;
                  return (
                    <line
                      key={step}
                      x1="36"
                      x2="864"
                      y1={y}
                      y2={y}
                      stroke="rgba(21, 28, 32, 0.09)"
                      strokeDasharray="6 7"
                    />
                  );
                })}
                {spectralMarkers.map((marker) => {
                  const x = 36 + ((marker.wavelength - 0.5) / (5.0 - 0.5)) * (900 - 72);
                  return (
                    <g key={marker.label}>
                      <line x1={x} x2={x} y1="28" y2="232" stroke="rgba(126, 82, 51, 0.28)" />
                      <text x={x + 8} y="48" fontSize="12" fill="rgba(126, 82, 51, 0.82)">
                        {marker.label}
                      </text>
                    </g>
                  );
                })}
                <polyline
                  fill="none"
                  stroke="#7e5233"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={buildPath("observed")}
                />
                <polyline
                  fill="none"
                  stroke="#1e5b63"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={buildPath("retrieved")}
                />
                <text x="36" y="250" fontSize="13" fill="rgba(21,28,32,0.56)">
                  0.5 μm
                </text>
                <text x="816" y="250" fontSize="13" fill="rgba(21,28,32,0.56)">
                  5.0 μm
                </text>
                <text x="36" y="20" fontSize="13" fill="rgba(21,28,32,0.56)">
                  normalized transit depth
                </text>
              </svg>
            </div>
          </article>

          <article className={styles.resultsColumn}>
            <div className={styles.metricInset}>
              <span className={styles.kicker}>Performance snapshot</span>
              <h2>Three numbers that anchor the whole pitch.</h2>
              <dl className={styles.metricList}>
                <div>
                  <dt>Training-phase best</dt>
                  <dd>{performance.bestTrainingVal.toFixed(6)}</dd>
                </div>
                <div>
                  <dt>Post-stop validation</dt>
                  <dd>{performance.validation.toFixed(6)}</dd>
                </div>
                <div>
                  <dt>Holdout</dt>
                  <dd>{performance.holdout.toFixed(6)}</dd>
                </div>
              </dl>
            </div>

            <div className={styles.metricInset}>
              <span className={styles.kicker}>Per-gas error</span>
              <div className={styles.barStack}>
                {performance.perGasRmse.map((entry) => (
                  <div key={entry.gas} className={styles.barRow}>
                    <div className={styles.barHeader}>
                      <span>{entry.gas}</span>
                      <span>{entry.value.toFixed(3)}</span>
                    </div>
                    <div className={styles.barTrack}>
                      <div
                        className={styles.barFill}
                        style={{ width: `${(entry.value / 0.42) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </article>
        </section>

        <section className={styles.gasSpread}>
          <div className={styles.sectionIntro}>
            <span className={styles.kicker}>Gas roster</span>
            <h2>The page keeps the chemistry explicit.</h2>
            <p>
              Every variant in this app has to keep the same scientific backbone, but this one treats the
              gases as editorial characters with specific jobs in the atmospheric interpretation.
            </p>
          </div>
          <div className={styles.gasGrid}>
            {gases.map((gas) => (
              <article key={gas.key} className={styles.gasCard}>
                <span className={styles.gasKey}>{gas.key}</span>
                <h3>{gas.label}</h3>
                <p>{gas.note}</p>
                <div className={styles.gasMeta}>
                  <span>{gas.role}</span>
                  <strong>{gas.highlight}</strong>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className={styles.modelSpread}>
          <div className={styles.sectionIntro}>
            <span className={styles.kicker}>Model lineup</span>
            <h2>Four routes, one benchmark frame.</h2>
            <p>
              The point of showing the lineup is not to drown the judges in ablation detail. It is to show
              that the custom quantum model is presented beside sensible reference models: the organizer
              baseline CNN, a Random Forest, a winner-style NSF implementation, and finally the hybrid
              quantum residual that the team is foregrounding.
            </p>
          </div>
          <div className={styles.modelRail}>
            {modelRoster.map((model, index) => (
              <article key={model.name} className={styles.modelCard}>
                <span className={styles.modelIndex}>0{index + 1}</span>
                <div>
                  <h3>{model.name}</h3>
                  <p className={styles.modelClass}>{model.className}</p>
                </div>
                <p>{model.summary}</p>
                <strong>{model.position}</strong>
              </article>
            ))}
          </div>
        </section>

        <section className={styles.architectureSpread}>
          <article className={styles.architectureNarrative}>
            <span className={styles.kicker}>Architecture note</span>
            <h2>A classical backbone with a quantum correction, not the other way around.</h2>
            <p>
              That sentence matters. The project reads as more scientifically mature when the quantum
              branch is described as a targeted residual mechanism layered onto a solid residual 1D encoder,
              rather than as a fragile end-to-end quantum stunt. The editorial framing makes that hierarchy
              legible fast.
            </p>
            <ul className={styles.architectureList}>
              {architecturePillars.map((pillar) => (
                <li key={pillar}>{pillar}</li>
              ))}
            </ul>
          </article>

          <div className={styles.architectureDiagram}>
            <div className={styles.track}>
              <span>Transmission spectra</span>
              <div />
            </div>
            <div className={styles.track}>
              <span>8 aux features</span>
              <div />
            </div>
            <div className={styles.mergeNode}>Residual encoder + fusion</div>
            <div className={styles.branchGrid}>
              <div className={styles.branchCard}>
                <strong>Classical head</strong>
                <p>Primary five-gas regression signal.</p>
              </div>
              <div className={styles.branchCard}>
                <strong>8-qubit branch</strong>
                <p>Per-target gated correction to the classical prediction.</p>
              </div>
            </div>
            <div className={styles.outputNode}>Final abundance vector: H2O / CO2 / CO / CH4 / NH3</div>
          </div>
        </section>

        <section className={styles.closingSpread}>
          <div className={styles.quoteBlock}>
            <p>
              “If the judges remember one thing, it should be that we framed biosignature detection as a
              systems problem, not a single-gas headline.”
            </p>
          </div>
          <div className={styles.talkTrack}>
            <span className={styles.kicker}>Two-minute talk track</span>
            <ol>
              <li>Lead with ADC2023 and the five-gas biosignature framing.</li>
              <li>Show the four-model ladder and the verified holdout number.</li>
              <li>Close on the hybrid quantum residual as the differentiator worth discussing.</li>
            </ol>
          </div>
        </section>
      </div>
    </main>
  );
}
