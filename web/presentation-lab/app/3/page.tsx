import type { CSSProperties } from "react";
import styles from "./page.module.css";
import { gases, modelRoster, performance, projectFacts, spectrumSeries, storyline } from "@/app/lib/project-data";

function buildSpectrumPath(field: "observed" | "retrieved") {
  const width = 420;
  const height = 128;
  const paddingX = 20;
  const paddingY = 18;
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

const observedPath = buildSpectrumPath("observed");
const retrievedPath = buildSpectrumPath("retrieved");

const gasOrbit = [
  { key: "H2O", angle: -76 },
  { key: "CO2", angle: -8 },
  { key: "CO", angle: 56 },
  { key: "CH4", angle: 132 },
  { key: "NH3", angle: 196 }
];

const noteBlocks = [
  {
    id: "northwest",
    title: "Field 01",
    heading: "Observational substrate",
    body: "Ariel Data Challenge 2023 is the atlas base layer: transmission spectra, auxiliary planetary context, and a benchmark that still feels scientifically honest in a hackathon demo.",
    lines: projectFacts.map((fact) => `${fact.label}: ${fact.value}`)
  },
  {
    id: "northeast",
    title: "Field 02",
    heading: "Verified checkpoint",
    body: "Three numbers carry the route. They sit near the radial center because the page is meant to orbit a single quantitative claim rather than a pile of widgets.",
    lines: [
      `training-phase validation: ${performance.bestTrainingVal.toFixed(6)}`,
      `re-evaluated validation: ${performance.validation.toFixed(6)}`,
      `re-evaluated holdout: ${performance.holdout.toFixed(6)}`
    ]
  },
  {
    id: "southwest",
    title: "Field 03",
    heading: "Model longitudes",
    body: "The comparison is expressed as a route map, not a leaderboard wall. The quantum model is framed as the newest longitude on top of established classical meridians.",
    lines: modelRoster.map((model) => `${model.name} • ${model.className}`)
  },
  {
    id: "southeast",
    title: "Field 04",
    heading: "Architecture arc",
    body: "Residual 1D encoder first. The 8-qubit branch arrives later as a gated correction on top of a strong classical estimate.",
    lines: storyline.map((step) => `${step.title}: ${step.detail}`)
  }
];

export default function VariantThreePage() {
  return (
    <main className={styles.page}>
      <div className={styles.sheet}>
        <header className={styles.topline}>
          <div>
            <p className={styles.kicker}>Solar Atlas / route 03 / annotated plate</p>
            <h1>Atlas of Five-Gas Biosignature Retrieval</h1>
          </div>
          <p className={styles.standfirst}>
            A planetary-map folio for a two-minute judge conversation: one central figure, marginal
            annotations, and a clear claim about joint abundance estimation for{" "}
            <strong>H2O, CO2, CO, CH4,</strong> and <strong>NH3</strong>.
          </p>
        </header>

        <section className={styles.plate}>
          {noteBlocks.map((block) => (
            <aside key={block.id} className={`${styles.marginNote} ${styles[block.id]}`}>
              <p className={styles.noteId}>{block.title}</p>
              <h2>{block.heading}</h2>
              <p className={styles.noteBody}>{block.body}</p>
              <ul>
                {block.lines.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </aside>
          ))}

          <figure className={styles.atlasFigure}>
            <div className={styles.outerRing} />
            <div className={styles.innerRing} />
            <div className={styles.equator} />
            <div className={styles.meridian} />
            <div className={styles.centerMedallion}>
              <span>holdout mRMSE</span>
              <strong>{performance.holdout.toFixed(4)}</strong>
              <small>best confirmed checkpoint</small>
            </div>

            <div className={styles.compass}>
              <span>N</span>
              <span>E</span>
              <span>S</span>
              <span>W</span>
            </div>

            <div className={styles.spectrumBand}>
              <svg viewBox="0 0 420 128" role="img" aria-label="Transmission spectrum inset">
                <rect x="0" y="0" width="420" height="128" rx="18" className={styles.spectrumFrame} />
                <polyline points={observedPath} className={styles.observedLine} />
                <polyline points={retrievedPath} className={styles.retrievedLine} />
                <text x="20" y="18">
                  observed
                </text>
                <text x="92" y="18">
                  retrieved
                </text>
                <text x="20" y="114">
                  0.5 μm
                </text>
                <text x="358" y="114">
                  5.0 μm
                </text>
              </svg>
            </div>

            {gasOrbit.map((gasMarker) => {
              const gas = gases.find((entry) => entry.key === gasMarker.key);
              if (!gas) {
                return null;
              }

              return (
                <div
                  key={gas.key}
                  className={styles.gasMarker}
                  style={{ "--angle": `${gasMarker.angle}deg` } as CSSProperties}
                >
                  <strong>{gas.key}</strong>
                  <span>{gas.label}</span>
                </div>
              );
            })}
          </figure>
        </section>

        <section className={styles.captionBand}>
          <div className={styles.captionBlock}>
            <p className={styles.bandId}>Legend A</p>
            <p>
              The circular plate organizes the whole story: dataset provenance to the northwest,
              verified metrics to the northeast, model longitudes to the southwest, and the
              classical-to-quantum architecture arc to the southeast.
            </p>
          </div>
          <div className={styles.captionBlock}>
            <p className={styles.bandId}>Legend B</p>
            <p>
              The hybrid route is explicit: residual 1D encoder, attention pooling, classical
              regression head, then an 8-qubit correction branch that refines rather than replaces
              the backbone.
            </p>
          </div>
          <div className={styles.captionBlock}>
            <p className={styles.bandId}>Legend C</p>
            <p>
              Scientific stance: the co-presence of these five gases is presented as biosignature
              evidence in context, not a standalone claim of life detection.
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}
