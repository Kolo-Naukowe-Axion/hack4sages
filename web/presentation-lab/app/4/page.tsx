import type { CSSProperties } from "react";
import {
  gases,
  modelRoster,
  performance,
  projectFacts,
  spectrumSeries,
  storyline
} from "@/app/lib/project-data";
import styles from "./page.module.css";

function spectrumPath(field: "observed" | "retrieved") {
  const width = 420;
  const height = 200;
  const paddingX = 20;
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

const metricSnapshot = [
  {
    label: "Training-phase val",
    value: performance.bestTrainingVal.toFixed(6),
    note: `best epoch ${performance.bestEpoch}`
  },
  {
    label: "Validation",
    value: performance.validation.toFixed(6),
    note: "re-evaluated checkpoint"
  },
  {
    label: "Holdout",
    value: performance.holdout.toFixed(6),
    note: `${performance.rows.toLocaleString()} rows`
  }
];

const orbitalStops = [
  { radius: 116, accent: "rgba(157, 233, 255, 0.26)" },
  { radius: 84, accent: "rgba(102, 198, 255, 0.32)" },
  { radius: 52, accent: "rgba(255, 214, 150, 0.34)" }
];

export default function VariantFourPage() {
  return (
    <main className={`page ${styles.page}`}>
      <div className={styles.aurora} aria-hidden="true" />
      <div className={styles.gridGlow} aria-hidden="true" />
      <div className="shell">
        <section className={styles.hero}>
          <article className={styles.leadPanel}>
            <span className={styles.kicker}>Glass Observatory / route 4</span>
            <h1 className={styles.title}>A premium lab deck for reading five-gas biosignatures.</h1>
            <p className={styles.summary}>
              This variant treats the demo like a transparent orbital laboratory: Ariel Data
              Challenge 2023 enters on one side, a hybrid quantum retrieval stack resolves the
              chemistry in the middle, and a five-gas biosignature briefing exits ready for a
              two-minute judge conversation.
            </p>

            <div className={styles.factRail}>
              {projectFacts.map((fact) => (
                <div key={fact.label} className={styles.factCard}>
                  <span>{fact.label}</span>
                  <strong>{fact.value}</strong>
                </div>
              ))}
            </div>
          </article>

          <aside className={styles.orbitPanel}>
            <div className={styles.orbitGlass}>
              <div className={styles.panelHeading}>
                <span>Observatory Metrics</span>
                <strong>Verified checkpoint snapshot</strong>
              </div>
              <div className={styles.orbitStage}>
                {orbitalStops.map((orbit) => (
                  <div
                    key={orbit.radius}
                    className={styles.orbitRing}
                    style={{
                      width: `${orbit.radius * 2}px`,
                      height: `${orbit.radius * 2}px`,
                      borderColor: orbit.accent
                    }}
                  />
                ))}
                <div className={styles.orbitCore}>
                  <span>8 qubits</span>
                  <strong>residual correction</strong>
                </div>
                {metricSnapshot.map((entry, index) => (
                  <div
                    key={entry.label}
                    className={styles.metricNode}
                    style={
                      {
                        "--x":
                          index === 0 ? "14%" : index === 1 ? "66%" : "38%",
                        "--y":
                          index === 0 ? "20%" : index === 1 ? "30%" : "74%"
                      } as CSSProperties
                    }
                  >
                    <span>{entry.label}</span>
                    <strong>{entry.value}</strong>
                    <small>{entry.note}</small>
                  </div>
                ))}
              </div>

              <div className={styles.spectrumShell}>
                <div className={styles.panelHeading}>
                  <span>Transmission spectrum</span>
                  <strong>Ariel-style observed vs retrieved profile</strong>
                </div>
                <svg viewBox="0 0 420 200" className={styles.spectrumPlot} role="img" aria-label="Observed and retrieved spectrum">
                  <defs>
                    <linearGradient id="gridFade" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="0%" stopColor="rgba(255,255,255,0.48)" />
                      <stop offset="100%" stopColor="rgba(255,255,255,0.08)" />
                    </linearGradient>
                  </defs>
                  <rect x="0" y="0" width="420" height="200" rx="24" fill="url(#gridFade)" />
                  {[0, 1, 2, 3].map((step) => {
                    const y = 22 + step * 44;
                    return (
                      <line
                        key={step}
                        x1="20"
                        x2="400"
                        y1={y}
                        y2={y}
                        stroke="rgba(216, 238, 255, 0.16)"
                        strokeDasharray="4 6"
                      />
                    );
                  })}
                  <polyline
                    fill="none"
                    stroke="rgba(255, 223, 182, 0.95)"
                    strokeWidth="3.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={spectrumPath("observed")}
                  />
                  <polyline
                    fill="none"
                    stroke="rgba(149, 232, 255, 0.95)"
                    strokeWidth="3.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={spectrumPath("retrieved")}
                  />
                  <text x="20" y="192" className={styles.plotLabel}>
                    0.5 μm
                  </text>
                  <text x="363" y="192" className={styles.plotLabel}>
                    5.0 μm
                  </text>
                </svg>
              </div>
            </div>
          </aside>
        </section>

        <section className={styles.contentGrid}>
          <article className={styles.glassCard}>
            <div className={styles.sectionHead}>
              <span>Five-gas observatory</span>
              <h2>Biosignatures are presented as a chemical constellation, not a single flare.</h2>
            </div>
            <div className={styles.gasGrid}>
              {gases.map((gas) => (
                <div key={gas.key} className={styles.gasCard}>
                  <div className={styles.gasHeader}>
                    <strong>{gas.key}</strong>
                    <span>{gas.label}</span>
                  </div>
                  <p>{gas.note}</p>
                  <small>{gas.highlight}</small>
                </div>
              ))}
            </div>
          </article>

          <article className={styles.glassCard}>
            <div className={styles.sectionHead}>
              <span>Model lineup</span>
              <h2>Classical references on the left, the hybrid quantum route on the right.</h2>
            </div>
            <div className={styles.modelStack}>
              {modelRoster.map((model, index) => (
                <div key={model.name} className={styles.modelCard}>
                  <div className={styles.modelIndex}>0{index + 1}</div>
                  <div>
                    <span>{model.position}</span>
                    <h3>{model.name}</h3>
                    <p>{model.className}</p>
                  </div>
                  <small>{model.summary}</small>
                </div>
              ))}
            </div>
          </article>
        </section>

        <section className={styles.architectureSection}>
          <div className={styles.sectionHead}>
            <span>Architecture walkthrough</span>
            <h2>Residual encoder first, quantum correction second, presentation-ready throughout.</h2>
          </div>

          <div className={styles.storyline}>
            {storyline.map((step, index) => (
              <article key={step.title} className={styles.storyCard}>
                <div className={styles.storyIndex}>0{index + 1}</div>
                <h3>{step.title}</h3>
                <p>{step.detail}</p>
              </article>
            ))}
          </div>

          <div className={styles.footnoteBar}>
            <span>Interpretation rule</span>
            <p>
              The route stays scientifically conservative: the joint abundance vector for H2O, CO2,
              CO, CH4, and NH3 is presented as biosignature context, not standalone proof of life.
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}
