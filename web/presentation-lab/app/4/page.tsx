import type { CSSProperties } from "react";
import styles from "./page.module.css";
import { gases, modelRoster, performance, projectFacts, spectrumSeries, storyline } from "@/app/lib/project-data";

function buildSpectrumPath(field: "observed" | "retrieved") {
  const width = 760;
  const height = 220;
  const paddingX = 34;
  const paddingY = 24;
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

const metricPanels = [
  {
    label: "Training-Phase Validation",
    value: performance.bestTrainingVal.toFixed(6),
    note: `best epoch ${performance.bestEpoch}`
  },
  {
    label: "Re-evaluated Validation",
    value: performance.validation.toFixed(6),
    note: "best_model.pt replay"
  },
  {
    label: "Re-evaluated Holdout",
    value: performance.holdout.toFixed(6),
    note: `${performance.rows.toLocaleString()} rows`
  }
];

const ceremonyBeats = [
  "Open on the problem: biosignatures are evaluated as a five-gas atmospheric composition, not a single attractive peak.",
  "Name the benchmark immediately: Ariel Data Challenge 2023 is the shared experimental substrate across all model families.",
  "Land on the architecture claim: a residual 1D encoder carries the main load, while the 8-qubit branch acts as a gated correction."
];

const observedPath = buildSpectrumPath("observed");
const retrievedPath = buildSpectrumPath("retrieved");

export default function VariantFourPage() {
  return (
    <main className={styles.page}>
      <div className={styles.constellation} aria-hidden="true" />
      <div className={styles.frame}>
        <header className={styles.marquee}>
          <p className={styles.pretitle}>Art Deco Observatory • Route 04 • ceremonial presentation mode</p>
          <div className={styles.titleBlock}>
            <span className={styles.sideLabel}>Axion Observatory Program</span>
            <h1>Five-Gas Biosignature Retrieval</h1>
            <span className={styles.sideLabel}>Hack-4-Sages 2026 • ETH Zurich orbit</span>
          </div>
          <p className={styles.subtitle}>
            A formal observatory-style presentation for scientist judges: Ariel transmission
            spectra enter the hall, a hybrid quantum retrieval stack interprets them, and the
            resulting abundance vector is framed with scientific restraint.
          </p>
          <div className={styles.metricTriptych}>
            {metricPanels.map((panel) => (
              <article key={panel.label} className={styles.metricPanel}>
                <span>{panel.label}</span>
                <strong>{panel.value}</strong>
                <small>{panel.note}</small>
              </article>
            ))}
          </div>
        </header>

        <div className={styles.divider} aria-hidden="true" />

        <section className={styles.axis}>
          <article className={styles.plaque}>
            <p className={styles.plaqueLabel}>Observatory Charter</p>
            <h2>Dataset and scientific framing</h2>
            <p className={styles.bodyCopy}>
              The route keeps the tone official: the project is presented as an atmospheric
              retrieval instrument trained on Ariel Data Challenge 2023, not as a speculative
              life-detection gimmick.
            </p>
            <dl className={styles.factList}>
              {projectFacts.map((fact) => (
                <div key={fact.label} className={styles.factRow}>
                  <dt>{fact.label}</dt>
                  <dd>{fact.value}</dd>
                </div>
              ))}
            </dl>
          </article>

          <section className={styles.celestialPlate} aria-label="Five-gas biosignature set">
            <div className={styles.innerRing} />
            <div className={styles.coreSeal}>
              <span>biosignature set</span>
              <strong>H2O • CO2 • CO • CH4 • NH3</strong>
            </div>
            {gases.map((gas, index) => (
              <article
                key={gas.key}
                className={styles.gasMarker}
                style={{ ["--marker-index" as const]: index } as CSSProperties}
              >
                <strong>{gas.key}</strong>
                <span>{gas.label}</span>
              </article>
            ))}
          </section>

          <article className={styles.plaque}>
            <p className={styles.plaqueLabel}>Interpretation Rule</p>
            <h2>Conservative by design</h2>
            <p className={styles.bodyCopy}>
              The combined abundance vector is used as biosignature context. No single gas on this
              page is treated as standalone proof of biology.
            </p>
            <div className={styles.emphasis}>
              <span>Formal claim</span>
              <p>
                The hybrid checkpoint’s holdout result of <strong>{performance.holdout.toFixed(6)}</strong>{" "}
                is presented as a retrieval-quality result, not a sensationalized discovery claim.
              </p>
            </div>
          </article>
        </section>

        <div className={styles.divider} aria-hidden="true" />

        <section className={styles.procession}>
          <article className={styles.processionBlock}>
            <p className={styles.plaqueLabel}>Model Procession</p>
            <h2>From baseline references to the hybrid quantum route</h2>
            <div className={styles.modelBands}>
              {modelRoster.map((model, index) => (
                <div key={model.name} className={styles.modelBand}>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <div>
                    <strong>{model.name}</strong>
                    <p>{model.className}</p>
                  </div>
                  <small>{model.summary}</small>
                </div>
              ))}
            </div>
          </article>

          <article className={styles.processionBlock}>
            <p className={styles.plaqueLabel}>Apparatus Walkthrough</p>
            <h2>Residual encoder first. Quantum correction second.</h2>
            <div className={styles.apparatus}>
              <div className={styles.quantumSeal}>
                <span>8 qubits</span>
                <strong>gated residual branch</strong>
              </div>
              <div className={styles.apparatusSteps}>
                {storyline.map((step, index) => (
                  <article key={step.title} className={styles.apparatusStep}>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <div>
                      <strong>{step.title}</strong>
                      <p>{step.detail}</p>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          </article>
        </section>

        <div className={styles.divider} aria-hidden="true" />

        <section className={styles.finale}>
          <article className={styles.spectrumPlaque}>
            <p className={styles.plaqueLabel}>Observation Plate</p>
            <h2>Transmission profile</h2>
            <svg viewBox="0 0 760 220" className={styles.spectrum} role="img" aria-label="Observed and retrieved transmission spectrum">
              <rect x="0" y="0" width="760" height="220" rx="22" className={styles.spectrumFrame} />
              {[0, 1, 2, 3].map((line) => {
                const y = 24 + line * 56;
                return <line key={line} x1="34" x2="726" y1={y} y2={y} className={styles.spectrumGrid} />;
              })}
              <polyline points={observedPath} className={styles.observedLine} />
              <polyline points={retrievedPath} className={styles.retrievedLine} />
              <text x="34" y="205">
                0.5 μm
              </text>
              <text x="674" y="205">
                5.0 μm
              </text>
            </svg>
          </article>

          <article className={styles.runbook}>
            <p className={styles.plaqueLabel}>Two-Minute Program</p>
            <h2>Speaking order for the judges</h2>
            <div className={styles.runbookList}>
              {ceremonyBeats.map((beat, index) => (
                <div key={beat} className={styles.runbookItem}>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <p>{beat}</p>
                </div>
              ))}
            </div>
          </article>
        </section>
      </div>
    </main>
  );
}
