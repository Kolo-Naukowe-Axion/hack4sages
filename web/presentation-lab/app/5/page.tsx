import styles from "./page.module.css";
import { gases, modelRoster, performance, projectFacts, spectrumSeries, storyline } from "@/app/lib/project-data";

const width = 860;
const height = 220;
const paddingX = 34;
const paddingY = 26;

const values = spectrumSeries.flatMap((point) => [point.observed, point.retrieved]);
const minX = spectrumSeries[0]?.wavelength ?? 0.5;
const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 5.0;
const minY = Math.min(...values) - 0.02;
const maxY = Math.max(...values) + 0.02;

function buildPath(field: "observed" | "retrieved") {
  return spectrumSeries
    .map((point) => {
      const x = paddingX + ((point.wavelength - minX) / (maxX - minX)) * (width - paddingX * 2);
      const y = height - paddingY - ((point[field] - minY) / (maxY - minY)) * (height - paddingY * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

const comparisonRows = [
  {
    system: "Organizer baseline CNN",
    family: "Classical convolutional baseline",
    role: "reference point",
    note: "Fast canonical starting line from the challenge ecosystem."
  },
  {
    system: "Random Forest",
    family: "Classical ensemble",
    role: "robust baseline",
    note: "Useful reality check against overclaiming neural or quantum novelty."
  },
  {
    system: "Winner-style NSF",
    family: "Challenge-winning family",
    role: "state-of-the-art reference",
    note: "Shows the project is benchmarked against a serious retrieval lineage."
  },
  {
    system: "Hybrid quantum residual",
    family: "Custom quantum ML",
    role: "team highlight",
    note: "Residual encoder plus 8-qubit correction branch, best verified holdout mRMSE 0.299376."
  }
];

const observedPath = buildPath("observed");
const retrievedPath = buildPath("retrieved");

export default function VariantFivePage() {
  return (
    <main className={styles.page}>
      <article className={styles.sheet}>
        <header className={styles.masthead}>
          <div className={styles.rail}>
            <span>AXION SCIENCE WEEKLY</span>
            <span>HACK-4-SAGES SPECIAL ISSUE</span>
            <span>ETH ZURICH ORBIT</span>
          </div>
          <div className={styles.nameplate}>
            <span className={styles.volume}>Vol. 01</span>
            <h1>The Biosignature Review</h1>
            <span className={styles.volume}>Route 05</span>
          </div>
          <div className={styles.rail}>
            <span>Transmission Spectroscopy Desk</span>
            <span>Five-Gas Abundance Edition</span>
            <span>Scientific Weekly Format</span>
          </div>
        </header>

        <section className={styles.leadGrid}>
          <aside className={styles.editionBox}>
            <p className={styles.label}>Dateline</p>
            <strong>Ariel Data Challenge 2023</strong>
            <p>
              The project begins with transmission spectra and auxiliary planetary context, then treats
              biosignature analysis as a five-abundance estimation problem.
            </p>
            <ul className={styles.bulletList}>
              <li>Targets: H2O, CO2, CO, CH4, NH3</li>
              <li>Benchmark split: 33,138 / 4,142 / 4,143</li>
              <li>Talkable in two minutes</li>
            </ul>
          </aside>

          <div className={styles.mainStory}>
            <p className={styles.kicker}>Front Page Science</p>
            <h2>From one transmission spectrum to a five-gas atmospheric argument.</h2>
            <p className={styles.dek}>
              This route is written like a broadsheet lead story: establish the dataset, frame the
              chemistry, show the benchmark lineup, and land on the one number the judges will
              remember. The result is serious, typographic, and intentionally unslick.
            </p>
            <p className={styles.byline}>By the ExoBiome desk, with emphasis on scientific restraint.</p>
          </div>

          <aside className={styles.scoreboard}>
            <p className={styles.label}>Verified checkpoint</p>
            <div className={styles.scoreValue}>{performance.holdout.toFixed(6)}</div>
            <p className={styles.scoreCaption}>Re-evaluated holdout mRMSE</p>
            <dl className={styles.metricList}>
              <div>
                <dt>Training best</dt>
                <dd>{performance.bestTrainingVal.toFixed(6)}</dd>
              </div>
              <div>
                <dt>Validation</dt>
                <dd>{performance.validation.toFixed(6)}</dd>
              </div>
              <div>
                <dt>Holdout rows</dt>
                <dd>{performance.rows.toLocaleString()}</dd>
              </div>
            </dl>
          </aside>
        </section>

        <section className={styles.bodyGrid}>
          <div className={styles.mainColumn}>
            <section className={styles.storySection}>
              <div className={styles.sectionHead}>
                <span>Lead analysis</span>
                <span>Page A1</span>
              </div>
              <div className={styles.columns}>
                <p>
                  The central move in this project is conceptual before it is technical: stop treating
                  biosignatures as a single dramatic gas and start treating them as a joint atmospheric
                  pattern. That is why the model predicts <strong>H2O</strong>, <strong>CO2</strong>,{" "}
                  <strong>CO</strong>, <strong>CH4</strong>, and <strong>NH3</strong> together, then
                  presents their co-presence as evidence to inspect rather than a headline to oversell.
                </p>
                <p>
                  The model story is equally disciplined. The classical backbone does most of the heavy
                  lifting: residual 1D spectral encoding, attention pooling, and a regression head that
                  already knows how to read the transmission signal. The quantum path is not a gimmick
                  replacement. It is an <strong>8-qubit correction branch</strong> that nudges the final
                  abundance vector where the classical system still leaves signal on the table.
                </p>
              </div>
            </section>

            <blockquote className={styles.pullQuote}>
              “The quantum block is presented as a correction layer, not as a theatrical bottleneck.
              That framing makes the whole story more credible.”
            </blockquote>

            <section className={styles.figureSection}>
              <div className={styles.sectionHead}>
                <span>Spectrum desk</span>
                <span>Page A2</span>
              </div>
              <h3>Observed transmission structure against the retrieved trend.</h3>
              <svg viewBox={`0 0 ${width} ${height}`} className={styles.figure} role="img" aria-label="Transmission spectrum">
                <rect x="0" y="0" width={width} height={height} className={styles.figureFrame} />
                {[0, 1, 2, 3].map((step) => {
                  const y = paddingY + step * ((height - paddingY * 2) / 3);
                  return (
                    <line
                      key={step}
                      x1={paddingX}
                      x2={width - paddingX}
                      y1={y}
                      y2={y}
                      className={styles.gridLine}
                    />
                  );
                })}
                <polyline points={observedPath} className={styles.observedLine} />
                <polyline points={retrievedPath} className={styles.retrievedLine} />
                <text x={paddingX} y={18}>
                  normalized transit depth
                </text>
                <text x={paddingX} y={height - 6}>
                  0.5 um
                </text>
                <text x={width - 70} y={height - 6}>
                  5.0 um
                </text>
              </svg>
              <p className={styles.caption}>
                Feature graphic. The page keeps a single figure and lets the typography carry the rest
                of the argument, as a newspaper would.
              </p>
            </section>

            <section className={styles.storySection}>
              <div className={styles.sectionHead}>
                <span>Chemistry briefing</span>
                <span>Page A3</span>
              </div>
              <div className={styles.gasTable}>
                {gases.map((gas) => (
                  <div key={gas.key} className={styles.gasRow}>
                    <strong>{gas.key}</strong>
                    <span>{gas.label}</span>
                    <p>{gas.highlight}</p>
                  </div>
                ))}
              </div>
            </section>
          </div>

          <aside className={styles.sidebar}>
            <section className={styles.sidebarBlock}>
              <p className={styles.label}>Notebook</p>
              <h3>Architecture in four desk notes.</h3>
              <ol className={styles.noteList}>
                {storyline.map((step) => (
                  <li key={step.title}>
                    <strong>{step.title}</strong>
                    <p>{step.detail}</p>
                  </li>
                ))}
              </ol>
            </section>

            <section className={styles.sidebarBlock}>
              <p className={styles.label}>Field facts</p>
              <dl className={styles.factList}>
                {projectFacts.map((fact) => (
                  <div key={fact.label}>
                    <dt>{fact.label}</dt>
                    <dd>{fact.value}</dd>
                  </div>
                ))}
              </dl>
            </section>

            <section className={styles.sidebarBlock}>
              <p className={styles.label}>Talk order</p>
              <ol className={styles.talkTrack}>
                <li>Open on the benchmark: Ariel Data Challenge 2023.</li>
                <li>Name the five gases and say why joint context matters.</li>
                <li>Show the model lineup to establish credibility.</li>
                <li>Land on the verified `0.299376` holdout mRMSE.</li>
              </ol>
            </section>
          </aside>
        </section>

        <section className={styles.comparisonSection}>
          <div className={styles.sectionHead}>
            <span>Comparative review</span>
            <span>Page A4</span>
          </div>
          <h3>The benchmark lineup in one full-width table.</h3>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>System</th>
                  <th>Family</th>
                  <th>Role</th>
                  <th>What the judges should hear</th>
                </tr>
              </thead>
              <tbody>
                {comparisonRows.map((row) => (
                  <tr key={row.system}>
                    <th>{row.system}</th>
                    <td>{row.family}</td>
                    <td>{row.role}</td>
                    <td>{row.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className={styles.caption}>
            The table is the key broadsheet device here: one sober comparison surface instead of four
            decorative feature cards.
          </p>
        </section>

        <footer className={styles.footer}>
          <p>
            Broadsheet review framing: dense by design, high on hierarchy, low on interface theatrics.
            That makes it fundamentally different from the other routes, which lean on spectacle,
            stagecraft, atlas composition, or instrument metaphors.
          </p>
          <div className={styles.rosterLine}>
            {modelRoster.map((model) => (
              <span key={model.name}>{model.name}</span>
            ))}
          </div>
        </footer>
      </article>
    </main>
  );
}
