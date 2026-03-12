import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig, Easing } from "remotion";
import { sans, mono } from "../lib/fonts";
import { COLORS } from "../lib/constants";
import { spectrumSeries } from "../lib/data";

const CHART = { left: 160, right: 100, top: 200, bottom: 120 };
const CW = 1920 - CHART.left - CHART.right;
const CH = 1080 - CHART.top - CHART.bottom;

// Key absorption features to annotate for physicists
const FEATURES = [
  { wl: 1.4, label: "H\u2082O", color: "#3b82f6" },
  { wl: 2.7, label: "CO\u2082 / H\u2082O", color: "#a855f7" },
  { wl: 4.3, label: "CO\u2082", color: "#ef4444" },
];

export const SpectrumScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const data = spectrumSeries;
  const minWl = data[0].wavelength;
  const maxWl = data[data.length - 1].wavelength;
  const allVals = data.flatMap((d) => [d.observed, d.retrieved]);
  const minY = Math.min(...allVals) - 0.02;
  const maxY = Math.max(...allVals) + 0.02;

  const toX = (wl: number) => CHART.left + ((wl - minWl) / (maxWl - minWl)) * CW;
  const toY = (v: number) => CHART.top + CH - ((v - minY) / (maxY - minY)) * CH;

  const sectionFade = interpolate(frame, [0, 0.8 * fps], [0, 1], { extrapolateRight: "clamp" });

  const titleFade = interpolate(frame, [0.3 * fps, 1.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const axesFade = interpolate(frame, [1 * fps, 2 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const observedProgress = interpolate(frame, [2 * fps, 9 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.quad),
  });

  const retrievedProgress = interpolate(frame, [5 * fps, 12 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.quad),
  });

  const legendFade = interpolate(frame, [9 * fps, 11 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const buildPolyline = (field: "observed" | "retrieved", progress: number) => {
    const count = Math.max(1, Math.floor(progress * data.length));
    return data
      .slice(0, count)
      .map((d) => `${toX(d.wavelength).toFixed(1)},${toY(d[field]).toFixed(1)}`)
      .join(" ");
  };

  const xTicks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      <div
        style={{
          position: "absolute",
          top: 50,
          left: 120,
          opacity: sectionFade,
          fontFamily: mono,
          fontSize: 14,
          letterSpacing: 4,
          textTransform: "uppercase",
          color: COLORS.textDim,
        }}
      >
        03 / the spectrum
      </div>

      <div
        style={{
          position: "absolute",
          top: 80,
          left: 120,
          right: 120,
          opacity: titleFade,
          fontFamily: sans,
          fontSize: 38,
          fontWeight: 800,
          letterSpacing: -2,
          color: COLORS.text,
          textTransform: "uppercase",
        }}
      >
        Absorption dips reveal the atmosphere&apos;s composition.
      </div>

      <svg width={1920} height={1080} style={{ position: "absolute" }}>
        {/* Grid */}
        <g opacity={axesFade}>
          {Array.from({ length: 5 }, (_, i) => {
            const y = CHART.top + (i * CH) / 4;
            return <line key={`y-${i}`} x1={CHART.left} y1={y} x2={CHART.left + CW} y2={y} stroke="rgba(255,255,255,0.08)" strokeDasharray="4 10" />;
          })}
          {xTicks.map((tick) => (
            <g key={tick}>
              <line x1={toX(tick)} y1={CHART.top} x2={toX(tick)} y2={CHART.top + CH} stroke="rgba(255,255,255,0.08)" strokeDasharray="4 10" />
              <text x={toX(tick)} y={CHART.top + CH + 30} fill={COLORS.textDim} fontSize={14} textAnchor="middle" fontFamily={mono}>
                {tick.toFixed(1)}
              </text>
            </g>
          ))}
          <text x={CHART.left + CW / 2} y={CHART.top + CH + 65} fill={COLORS.textDim} fontSize={16} textAnchor="middle" fontFamily={mono}>
            wavelength ({"\u03bcm"})
          </text>
          <text x={55} y={CHART.top + CH / 2} fill={COLORS.textDim} fontSize={16} textAnchor="middle" fontFamily={mono} transform={`rotate(-90, 55, ${CHART.top + CH / 2})`}>
            normalized transit depth
          </text>
        </g>

        {/* Observed */}
        {observedProgress > 0 && (
          <polyline points={buildPolyline("observed", observedProgress)} fill="none" stroke={COLORS.observed} strokeWidth={4} strokeLinecap="round" strokeLinejoin="round" />
        )}

        {/* Retrieved */}
        {retrievedProgress > 0 && (
          <polyline points={buildPolyline("retrieved", retrievedProgress)} fill="none" stroke={COLORS.retrieved} strokeWidth={4} strokeLinecap="round" strokeLinejoin="round" />
        )}

        {/* Absorption feature annotations */}
        {FEATURES.map((feat) => {
          const wlProgress = (feat.wl - minWl) / (maxWl - minWl);
          const appearsAt = 2 * fps + wlProgress * 7 * fps + 1 * fps;
          const featFade = interpolate(frame, [appearsAt, appearsAt + 1 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });

          const x = toX(feat.wl);

          return (
            <g key={feat.label} opacity={featFade}>
              <line x1={x} y1={CHART.top + 20} x2={x} y2={CHART.top + CH - 20} stroke={feat.color} strokeDasharray="5 4" strokeWidth={1.5} strokeOpacity={0.5} />
              <text x={x} y={CHART.top + 10} fill={feat.color} fontSize={15} fontWeight={600} textAnchor="middle" fontFamily={mono}>
                {feat.label}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div
        style={{
          position: "absolute",
          top: CHART.top - 10,
          right: CHART.right + 20,
          opacity: legendFade,
          display: "flex",
          gap: 30,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 24, height: 3, backgroundColor: COLORS.observed, borderRadius: 2 }} />
          <span style={{ fontFamily: mono, fontSize: 14, color: COLORS.textDim }}>observed</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 24, height: 3, backgroundColor: COLORS.retrieved, borderRadius: 2 }} />
          <span style={{ fontFamily: mono, fontSize: 14, color: COLORS.textDim }}>model retrieval</span>
        </div>
      </div>

      {/* Bottom note */}
      <div
        style={{
          position: "absolute",
          bottom: 30,
          left: 120,
          opacity: legendFade,
          fontFamily: sans,
          fontSize: 17,
          color: COLORS.textDim,
        }}
      >
        52 spectral bins &middot; 0.5 &ndash; 5.0 {"\u03bcm"} &middot; Ariel Data Challenge 2023
      </div>
    </AbsoluteFill>
  );
};
