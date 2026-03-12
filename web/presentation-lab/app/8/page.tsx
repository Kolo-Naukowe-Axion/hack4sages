import styles from "./page.module.css";
import { gases, modelRoster, performance, spectrumSeries, storyline } from "@/app/lib/project-data";

function curve(field: "observed" | "retrieved") {
  const width = 560;
  const height = 190;
  const paddingX = 22;
  const paddingY = 20;
  const minX = spectrumSeries[0]?.wavelength ?? 0.5;
  const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 5;
  const values = spectrumSeries.flatMap((entry) => [entry.observed, entry.retrieved]);
  const minY = Math.min(...values) - 0.02;
  const maxY = Math.max(...values) + 0.02;

  return spectrumSeries
    .map((entry) => {
      const x = paddingX + ((entry.wavelength - minX) / (maxX - minX)) * (width - paddingX * 2);
      const y =
        height - paddingY - ((entry[field] - minY) / (maxY - minY)) * (height - paddingY * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

const observed = curve("observed");
const retrieved = curve("retrieved");

export default function VariantEightPage() {
  return (
    <main className={styles.page}>
      <div className={styles.archive}>
        <aside className={styles.spine}>
          <span>Archive</span>
          <span>Biosignatures</span>
          <span>Route 08</span>
        </aside>

        <div className={styles.content}>
          <header className={styles.header}>
            <div className={styles.plaque}>
              <p className={styles.kicker}>Cabinet of Specimens / Judge Edition</p>
              <h1>Five Gases in One Archive Drawer</h1>
              <p className={styles.lead}>
                This route treats the project like a natural-history cabinet: measured labels,
                specimen drawers, and one central curator note about what the model actually does and
                what it does not claim.
              </p>
            </div>
            <div className={styles.score}>
              <span>Verified holdout checkpoint</span>
              <strong>{performance.holdout.toFixed(6)}</strong>
              <p>with validation snapshots at {performance.bestTrainingVal.toFixed(6)} and {performance.validation.toFixed(6)}</p>
            </div>
          </header>

          <section className={styles.cabinet}>
            {gases.map((gas) => (
              <article key={gas.key} className={styles.drawer}>
                <div className={styles.tab}>{gas.key}</div>
                <h2>{gas.label}</h2>
                <p>{gas.note}</p>
                <small>{gas.highlight}</small>
              </article>
            ))}

            <article className={styles.centerpiece}>
              <span className={styles.centerLabel}>Specimen label</span>
              <h2>Ariel Data Challenge 2023</h2>
              <p>
                The cabinet holds a single retrieval problem: infer H2O, CO2, CO, CH4, and NH3 from
                transmission spectroscopy plus auxiliary context, then discuss co-presence as
                biosignature evidence instead of a single-gas headline.
              </p>
            </article>
          </section>

          <section className={styles.lower}>
            <article className={styles.column}>
              <div className={styles.block}>
                <p className={styles.kicker}>Lineage shelf</p>
                <ul className={styles.modelShelf}>
                  {modelRoster.map((model) => (
                    <li key={model.name}>
                      <strong>{model.name}</strong>
                      <span>{model.className}</span>
                      <p>{model.summary}</p>
                    </li>
                  ))}
                </ul>
              </div>

              <div className={styles.block}>
                <p className={styles.kicker}>Curator note</p>
                <p className={styles.note}>
                  The hybrid quantum residual is the featured object here because it preserves a strong
                  classical spine and uses the 8-qubit branch as a correction layer, which is a much
                  more defensible scientific story than pretending the quantum block does everything.
                </p>
              </div>
            </article>

            <article className={styles.column}>
              <div className={styles.block}>
                <p className={styles.kicker}>Retrieval slip</p>
                <div className={styles.slip}>
                  {storyline.map((step, index) => (
                    <div key={step.title} className={styles.slipRow}>
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      <div>
                        <strong>{step.title}</strong>
                        <p>{step.detail}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className={styles.block}>
                <p className={styles.kicker}>Observation strip</p>
                <svg viewBox="0 0 560 190" className={styles.spectrum} role="img" aria-label="Transmission spectrum strip">
                  <rect x="0" y="0" width="560" height="190" className={styles.frame} />
                  <polyline points={observed} className={styles.observed} />
                  <polyline points={retrieved} className={styles.retrieved} />
                  <text x="22" y="26">observed</text>
                  <text x="96" y="26">retrieved</text>
                  <text x="22" y="174">0.5 um</text>
                  <text x="492" y="174">5.0 um</text>
                </svg>
              </div>
            </article>
          </section>
        </div>
      </div>
    </main>
  );
}
