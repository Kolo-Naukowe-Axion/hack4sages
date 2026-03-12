import styles from "./page.module.css";
import { gases, modelRoster, performance, projectFacts, spectrumSeries, storyline } from "@/app/lib/project-data";

const chartWidth = 920;
const chartHeight = 260;
const paddingX = 24;
const paddingY = 26;
const values = spectrumSeries.flatMap((point) => [point.observed, point.retrieved]);
const minY = Math.min(...values) - 0.02;
const maxY = Math.max(...values) + 0.02;
const maxRmse = Math.max(...performance.perGasRmse.map((entry) => entry.value));

function projectPoint(wavelength: number, value: number) {
  const minX = spectrumSeries[0]?.wavelength ?? 0.5;
  const maxX = spectrumSeries[spectrumSeries.length - 1]?.wavelength ?? 5;
  const x = paddingX + ((wavelength - minX) / (maxX - minX)) * (chartWidth - paddingX * 2);
  const y =
    chartHeight - paddingY - ((value - minY) / (maxY - minY)) * (chartHeight - paddingY * 2);
  return `${x.toFixed(1)},${y.toFixed(1)}`;
}

const observedPath = spectrumSeries.map((point) => projectPoint(point.wavelength, point.observed)).join(" ");
const retrievedPath = spectrumSeries.map((point) => projectPoint(point.wavelength, point.retrieved)).join(" ");

const runbook = [
  "1. State the benchmark first: Ariel Data Challenge 2023 transmission spectroscopy.",
  "2. Point at the score line: 0.299376 holdout mRMSE, 0.293614 validation, 0.290811 training-phase best validation.",
  "3. Name the target set explicitly: H2O, CO2, CO, CH4, NH3.",
  "4. Explain the architecture in one line: residual 1D encoder plus an 8-qubit correction branch.",
  "5. Close with the scientific caveat: combined atmospheric context, not standalone proof of life."
];

export default function VariantTwoPage() {
  return (
    <main className={styles.page}>
      <div className={styles.scanlines} aria-hidden="true" />
      <div className={styles.viewport}>
        <header className={styles.topline}>
          <span>CRT SPECTROGRAPH // ROUTE 02 // OBSERVATORY TERMINAL</span>
          <span>LINK: STABLE</span>
          <span>MODE: BIOSIGNATURE REVIEW</span>
        </header>

        <section className={styles.block}>
          <div className={styles.promptLine}>
            <span className={styles.prompt}>$</span>
            <span>boot --dataset adc2023 --targets h2o,co2,co,ch4,nh3 --mode retrieval</span>
          </div>
          <div className={styles.bootGrid}>
            <div>
              <p className={styles.statusLine}>[ok] core benchmark :: Ariel Data Challenge 2023</p>
              <p className={styles.statusLine}>[ok] ingest :: 52 spectral bins x 4 channels + 8 auxiliary features</p>
              <p className={styles.statusLine}>[ok] hybrid path :: residual 1D encoder + 8-qubit correction branch</p>
              <h1 className={styles.title}>FIVE-GAS RETRIEVAL THROUGH A PHOSPHOR INSTRUMENT LENS</h1>
              <p className={styles.lead}>
                This route behaves like an observatory terminal instead of a website. It turns your
                project into a readable command session: load the dataset, inspect the spectral fit,
                check the model ladder, and end on the verified mRMSE.
              </p>
            </div>
            <div className={styles.metricStack}>
              <div className={styles.metricRow}>
                <span>TRAINING_PHASE_BEST_VAL</span>
                <strong>{performance.bestTrainingVal.toFixed(6)}</strong>
              </div>
              <div className={styles.metricRow}>
                <span>VALIDATION_REPLAY</span>
                <strong>{performance.validation.toFixed(6)}</strong>
              </div>
              <div className={styles.metricRow}>
                <span>HOLDOUT_VERIFIED</span>
                <strong>{performance.holdout.toFixed(6)}</strong>
              </div>
              <div className={styles.metricRow}>
                <span>ROWS</span>
                <strong>{performance.rows.toLocaleString()}</strong>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.block}>
          <div className={styles.promptLine}>
            <span className={styles.prompt}>$</span>
            <span>scope --channel transmission --overlay observed,retrieved</span>
          </div>
          <div className={styles.scopeLayout}>
            <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className={styles.scope} role="img" aria-label="Transmission spectrum overlay">
              <rect x="0" y="0" width={chartWidth} height={chartHeight} className={styles.scopeFrame} />
              {Array.from({ length: 6 }, (_, index) => {
                const y = paddingY + index * ((chartHeight - paddingY * 2) / 5);
                return (
                  <line
                    key={index}
                    x1={paddingX}
                    x2={chartWidth - paddingX}
                    y1={y}
                    y2={y}
                    className={styles.scopeLine}
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
                    y1={paddingY}
                    y2={chartHeight - paddingY}
                    className={styles.scopeMarker}
                  />
                );
              })}
              <polyline points={observedPath} className={styles.observedTrace} />
              <polyline points={retrievedPath} className={styles.retrievedTrace} />
              <text x="24" y="18">OBSERVED</text>
              <text x="124" y="18">RETRIEVED</text>
              <text x="24" y="244">0.5 UM</text>
              <text x="836" y="244">5.0 UM</text>
            </svg>

            <div className={styles.sideLog}>
              <p>&gt; spectral overlay confirms retrieval shape across transmission band</p>
              <p>&gt; five targets are inferred jointly, not through five separate demos</p>
              <p>&gt; display note: biosignature evidence requires context, not isolated peaks</p>
              <div className={styles.warning}>[warn] methane without CO / CO2 context would be an overclaim</div>
            </div>
          </div>
        </section>

        <section className={styles.block}>
          <div className={styles.promptLine}>
            <span className={styles.prompt}>$</span>
            <span>print --gas-matrix --comparison --architecture</span>
          </div>
          <div className={styles.matrix}>
            <div className={styles.rowHead}>TARGET_SET</div>
            <div className={styles.rowBody}>
              {gases.map((gas) => (
                <div key={gas.key} className={styles.inlineCell}>
                  <strong>{gas.key}</strong>
                  <span>{gas.label}</span>
                  <small>{gas.highlight}</small>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.terminalTable}>
            <div className={styles.tableHeader}>
              <span>MODEL</span>
              <span>FAMILY</span>
              <span>ROLE</span>
            </div>
            {modelRoster.map((model) => (
              <div key={model.name} className={styles.tableRow}>
                <strong>{model.name}</strong>
                <span>{model.className}</span>
                <span>{model.summary}</span>
              </div>
            ))}
          </div>

          <div className={styles.promptDump}>
            {storyline.map((step, index) => (
              <div key={step.title} className={styles.dumpLine}>
                <span>{`0${index + 1}`}</span>
                <strong>{step.title}</strong>
                <p>{step.detail}</p>
              </div>
            ))}
          </div>
        </section>

        <section className={styles.block}>
          <div className={styles.promptLine}>
            <span className={styles.prompt}>$</span>
            <span>report --holdout-rmse --facts --demo-runbook</span>
          </div>

          <div className={styles.reportGrid}>
            <div>
              <div className={styles.factHeader}>FACTS</div>
              <div className={styles.factList}>
                {projectFacts.map((fact) => (
                  <div key={fact.label} className={styles.factLine}>
                    <span>{fact.label}</span>
                    <strong>{fact.value}</strong>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className={styles.factHeader}>PER_GAS_RMSE</div>
              <div className={styles.rmseList}>
                {performance.perGasRmse.map((entry) => (
                  <div key={entry.gas} className={styles.rmseRow}>
                    <div className={styles.rmseMeta}>
                      <span>{entry.gas}</span>
                      <strong>{entry.value.toFixed(3)}</strong>
                    </div>
                    <div className={styles.rmseTrack}>
                      <div className={styles.rmseFill} style={{ width: `${(entry.value / maxRmse) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className={styles.factHeader}>DEMO_SEQUENCE</div>
              <ol className={styles.runbook}>
                {runbook.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ol>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
