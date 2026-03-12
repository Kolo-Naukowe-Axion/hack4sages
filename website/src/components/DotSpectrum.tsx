import { spectralMarkers } from "@/data/presentation-data";

export function DotSpectrum() {
  const points = [0.06, 0.12, 0.18, 0.24, 0.29, 0.34, 0.4, 0.46, 0.52, 0.6, 0.67, 0.73, 0.81, 0.88, 0.94];

  return (
    <article className="spectrum-shell">
      <div className="chart-meta">
        <div>
          <p className="chart-title">Transmission signature</p>
          <p className="chart-note">placeholder spectral windows in dot form</p>
        </div>
      </div>

      <div className="spectrum-track">
        {points.map((point, index) => (
          <span
            key={index}
            className="spectrum-dot"
            style={{
              left: `${point * 100}%`,
              top: `${40 + Math.sin(index * 0.8) * 18}px`,
            }}
          />
        ))}

        {spectralMarkers.map((marker) => (
          <div
            key={marker.gas}
            className="spectrum-window"
            style={{
              left: `${marker.x * 100}%`,
              width: `${marker.width * 100}%`,
              borderColor: "rgba(17, 15, 11, 0.18)",
              backgroundColor: "rgba(17, 15, 11, 0.03)",
            }}
          >
            <span>{marker.gas}</span>
          </div>
        ))}
      </div>
    </article>
  );
}
