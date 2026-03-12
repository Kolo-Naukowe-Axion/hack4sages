import styles from "./page.module.css";
import { gases, modelRoster, performance, projectFacts, spectrumSeries, storyline } from "@/app/lib/project-data";

function spectrumPath(field: "observed" | "retrieved") {
  const width = 760;
  const height = 220;
  const paddingX = 32;
  const paddingY = 24;
  const minX = spectrumSeries[0]?.wavelength ?? 0.5;
  const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 5.0;
  const values = spectrumSeries.flatMap((point) => [point.observed, point.retrieved]);
  const minY = Math.min(...values) - 0.02;
  const maxY = Math.max(...values) + 0.02;

  return spectrumSeries
    .map((point) => {
      const x = paddingX + ((point.wavelength - minX) / (maxX - minX)) * (width - paddingX * 2);
      const y =
        height - paddingY - ((point[field] - minY) / (maxY - minY)) * (height - paddingY * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

const blueprintNotes = [
  {
    id: "A1",
    title: "Input sheet",
    detail: "Ariel Data Challenge 2023 spectra and 8 auxiliary features enter as a single retrieval package."
  },
  {
    id: "B2",
    title: "Encoder spine",
    detail: "Residual 1D spectral encoding and attention pooling establish the classical baseline estimate."
  },
  {
    id: "C3",
    title: "Quantum residual",
    detail: "An 8-qubit correction branch refines the five-gas abundance vector rather than replacing the backbone."
  },
  {
    id: "D4",
    title: "Interpretation rule",
    detail: "H2O, CO2, CO, CH4, and NH3 are read jointly as biosignature context, not standalone proof of life."
  }
];

const observed = spectrumPath("observed");
const retrieved = spectrumPath("retrieved");

export default function VariantSevenPage() {
  return (
    <main className={styles.page}>
      <div className={styles.sheet}>
        <header className={styles.header}>
          <div>
            <p className={styles.serial}>Route 07 / Blueprint Roll / Technical Edition</p>
            <h1>ExoBiome Retrieval Blueprint</h1>
          </div>
          <div className={styles.metricRibbon}>
            <div>
              <span>Training val</span>
              <strong>{performance.bestTrainingVal.toFixed(6)}</strong>
            </div>
            <div>
              <span>Validation</span>
              <strong>{performance.validation.toFixed(6)}</strong>
            </div>
            <div>
              <span>Holdout</span>
              <strong>{performance.holdout.toFixed(6)}</strong>
            </div>
          </div>
        </header>

        <section className={styles.board}>
          <aside className={styles.margin}>
            <div className={styles.marginBlock}>
              <span className={styles.label}>Dataset</span>
              <strong>Ariel Data Challenge 2023</strong>
            </div>
            <div className={styles.marginBlock}>
              <span className={styles.label}>Targets</span>
              <strong>{gases.map((gas) => gas.key).join(" / ")}</strong>
            </div>
            <div className={styles.marginBlock}>
              <span className={styles.label}>Fact sheet</span>
              <ul>
                {projectFacts.map((fact) => (
                  <li key={fact.label}>
                    <span>{fact.label}</span>
                    <strong>{fact.value}</strong>
                  </li>
                ))}
              </ul>
            </div>
          </aside>

          <article className={styles.drawing}>
            <div className={styles.titleBlock}>
              <p>
                This route is not a dashboard. It is a single engineering drawing for judges who want
                the project explained as a mechanism: source data, signal handling, model family,
                checkpoint evidence, and interpretation limits.
              </p>
            </div>

            <svg viewBox="0 0 960 560" className={styles.svg} role="img" aria-label="Blueprint diagram">
              <defs>
                <pattern id="bp-grid" width="24" height="24" patternUnits="userSpaceOnUse">
                  <path d="M 24 0 L 0 0 0 24" fill="none" stroke="rgba(121,238,255,0.10)" strokeWidth="1" />
                </pattern>
              </defs>
              <rect x="0" y="0" width="960" height="560" fill="url(#bp-grid)" />

              <rect x="54" y="72" width="172" height="76" className={styles.block} />
              <rect x="304" y="72" width="204" height="76" className={styles.block} />
              <rect x="586" y="72" width="146" height="76" className={styles.block} />
              <rect x="790" y="72" width="116" height="76" className={styles.block} />
              <path d="M226 110 H304" className={styles.link} />
              <path d="M508 110 H586" className={styles.link} />
              <path d="M732 110 H790" className={styles.link} />

              <text x="72" y="104">spectra + aux</text>
              <text x="338" y="104">residual encoder</text>
              <text x="617" y="104">8-qubit branch</text>
              <text x="823" y="104">5 gases</text>

              <rect x="56" y="208" width="848" height="260" className={styles.frame} />
              {[0, 1, 2, 3].map((row) => {
                const y = 230 + row * 48;
                return <line key={row} x1="82" x2="878" y1={y} y2={y} className={styles.grid} />;
              })}
              <polyline points={observed} className={styles.observed} />
              <polyline points={retrieved} className={styles.retrieved} />
              <text x="82" y="220">transmission spectrum blueprint</text>
              <text x="82" y="452">0.5 um</text>
              <text x="826" y="452">5.0 um</text>

              <circle cx="184" cy="318" r="7" className={styles.dot} />
              <path d="M184 318 H118 V502" className={styles.callout} />
              <text x="54" y="524">A1 input band</text>

              <circle cx="414" cy="332" r="7" className={styles.dot} />
              <path d="M414 332 V494 H336" className={styles.callout} />
              <text x="246" y="524">B2 encoder baseline</text>

              <circle cx="644" cy="304" r="7" className={styles.dot} />
              <path d="M644 304 V494 H586" className={styles.callout} />
              <text x="506" y="524">C3 8-qubit residual</text>

              <circle cx="818" cy="290" r="7" className={styles.dot} />
              <path d="M818 290 V494 H764" className={styles.callout} />
              <text x="692" y="524">D4 five-gas reading</text>
            </svg>

            <div className={styles.noteRow}>
              {blueprintNotes.map((note) => (
                <article key={note.id} className={styles.note}>
                  <span>{note.id}</span>
                  <strong>{note.title}</strong>
                  <p>{note.detail}</p>
                </article>
              ))}
            </div>
          </article>
        </section>

        <section className={styles.lower}>
          <article className={styles.lineage}>
            <p className={styles.label}>Model lineage</p>
            <ol>
              {modelRoster.map((model) => (
                <li key={model.name}>
                  <strong>{model.name}</strong>
                  <span>{model.className}</span>
                  <p>{model.summary}</p>
                </li>
              ))}
            </ol>
          </article>

          <article className={styles.runbook}>
            <p className={styles.label}>Two-minute readout</p>
            <div className={styles.steps}>
              {storyline.map((step, index) => (
                <div key={step.title} className={styles.step}>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <div>
                    <strong>{step.title}</strong>
                    <p>{step.detail}</p>
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
