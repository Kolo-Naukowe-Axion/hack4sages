import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig, Easing } from "remotion";
import { sans, mono } from "../lib/fonts";
import { COLORS } from "../lib/constants";

const BOTTLENECK_FACTS = [
  { icon: "~50", label: "exoplanets with atmospheric detections so far" },
  { icon: "100+ hrs", label: "of JWST time per single detailed spectrum" },
  { icon: "~10k", label: "synthetic spectra available for training" },
];

export const RetrievalScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const sectionFade = interpolate(frame, [0, 0.8 * fps], [0, 1], { extrapolateRight: "clamp" });

  const headingFade = interpolate(frame, [0.3 * fps, 1.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const descFade = interpolate(frame, [1.5 * fps, 3 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const problemFade = interpolate(frame, [5.5 * fps, 7 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const questionFade = interpolate(frame, [9 * fps, 11 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: COLORS.bg,
        padding: "80px 120px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
      }}
    >
      <div style={{ opacity: sectionFade, fontFamily: mono, fontSize: 14, letterSpacing: 4, textTransform: "uppercase", color: COLORS.textDim, marginBottom: 20 }}>
        04 / the real bottleneck
      </div>

      <div style={{ opacity: headingFade, fontFamily: sans, fontSize: 68, fontWeight: 800, lineHeight: 0.92, letterSpacing: -3, color: COLORS.text, textTransform: "uppercase", maxWidth: 900, marginBottom: 30 }}>
        Data is scarce.
        <br />
        Every spectrum
        <br />
        counts.
      </div>

      <div
        style={{
          opacity: descFade,
          fontFamily: sans,
          fontSize: 22,
          fontWeight: 400,
          color: COLORS.textMuted,
          lineHeight: 1.6,
          maxWidth: 900,
          marginBottom: 60,
        }}
      >
        Atmospheric retrieval is an inverse problem: recover gas abundances from a noisy spectrum.
        But the fundamental challenge in exoplanet science isn't the algorithm {"\u2014"} it's the
        amount of data. Telescope time is expensive, real observations are rare, and synthetic
        datasets are small by ML standards.
      </div>

      {/* Bottleneck facts */}
      <div style={{ display: "flex", gap: 50 }}>
        {BOTTLENECK_FACTS.map((fact, i) => {
          const delay = 3.5 * fps + i * 15;
          const factFade = interpolate(frame, [delay, delay + 0.8 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.quad),
          });

          return (
            <div
              key={fact.label}
              style={{
                opacity: factFade,
                padding: "28px 36px",
                border: `1px solid ${COLORS.border}`,
                borderRadius: 8,
                flex: 1,
              }}
            >
              <div style={{ fontFamily: sans, fontSize: 44, fontWeight: 800, color: COLORS.accent, lineHeight: 1 }}>
                {fact.icon}
              </div>
              <div style={{ fontFamily: sans, fontSize: 16, color: COLORS.textMuted, marginTop: 10, lineHeight: 1.5 }}>
                {fact.label}
              </div>
            </div>
          );
        })}
      </div>

      {/* Problem statement */}
      <div
        style={{
          opacity: problemFade,
          fontFamily: sans,
          fontSize: 24,
          fontWeight: 700,
          color: COLORS.text,
          textTransform: "uppercase",
          letterSpacing: -1,
          marginTop: 50,
          paddingTop: 24,
          borderTop: `1px solid ${COLORS.border}`,
        }}
      >
        Models that need massive datasets won't work here.
      </div>

      <div
        style={{
          opacity: questionFade,
          fontFamily: sans,
          fontSize: 20,
          fontWeight: 400,
          color: COLORS.textMuted,
          marginTop: 12,
          lineHeight: 1.6,
        }}
      >
        We need an approach that learns faster and generalizes better from limited training data.
      </div>
    </AbsoluteFill>
  );
};
