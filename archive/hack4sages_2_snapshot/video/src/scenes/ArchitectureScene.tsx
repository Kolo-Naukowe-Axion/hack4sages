import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig, Easing } from "remotion";
import { sans, mono } from "../lib/fonts";
import { COLORS } from "../lib/constants";

const STEPS = [
  {
    num: "01",
    title: "Classical encoder",
    detail: "A residual 1D convolutional network compresses the spectrum into a compact latent representation.",
    highlight: false,
  },
  {
    num: "02",
    title: "Attention pooling",
    detail: "Learns which spectral regions matter most \u2014 focusing compute on absorption features, not noise.",
    highlight: false,
  },
  {
    num: "03",
    title: "8-qubit quantum branch",
    detail: "A parameterized quantum circuit on real IQM hardware applies a learned correction. Quantum feature maps capture correlations that classical networks need far more data to learn.",
    highlight: true,
  },
  {
    num: "04",
    title: "Joint five-gas output",
    detail: "Predicts all five abundances simultaneously \u2014 exploiting the physical correlations between gases.",
    highlight: false,
  },
];

export const ArchitectureScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const sectionFade = interpolate(frame, [0, 0.8 * fps], [0, 1], { extrapolateRight: "clamp" });

  const headingFade = interpolate(frame, [0.3 * fps, 1.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const wiresFade = interpolate(frame, [8 * fps, 10 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const noteFade = interpolate(frame, [9 * fps, 11 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: COLORS.bg,
        padding: "70px 120px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
      }}
    >
      <div style={{ opacity: sectionFade, fontFamily: mono, fontSize: 14, letterSpacing: 4, textTransform: "uppercase", color: COLORS.textDim, marginBottom: 20 }}>
        05 / why quantum helps
      </div>

      <div style={{ opacity: headingFade, fontFamily: sans, fontSize: 56, fontWeight: 800, lineHeight: 0.92, letterSpacing: -3, color: COLORS.text, textTransform: "uppercase", maxWidth: 900, marginBottom: 45 }}>
        Learn more
        <br />
        from less data.
      </div>

      {/* Steps */}
      <div style={{ display: "flex", flexDirection: "column" }}>
        {STEPS.map((step, i) => {
          const delay = 2 * fps + i * 20;
          const stepFade = interpolate(frame, [delay, delay + 0.8 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.quad),
          });

          return (
            <div
              key={step.num}
              style={{
                opacity: stepFade,
                display: "flex",
                gap: 28,
                alignItems: "flex-start",
                padding: "20px 0",
                borderTop: `1px solid ${COLORS.border}`,
              }}
            >
              <div style={{ fontFamily: sans, fontSize: 44, fontWeight: 800, lineHeight: 0.9, color: step.highlight ? COLORS.accent : COLORS.textFaint, minWidth: 70 }}>
                {step.num}
              </div>
              <div>
                <div style={{ fontFamily: sans, fontSize: 24, fontWeight: 700, textTransform: "uppercase", letterSpacing: -0.5, color: step.highlight ? COLORS.accent : COLORS.text, marginBottom: 4 }}>
                  {step.title}
                </div>
                <div style={{ fontFamily: sans, fontSize: 17, fontWeight: 400, color: COLORS.textMuted, lineHeight: 1.6, maxWidth: 900 }}>
                  {step.detail}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* 8 qubit wire lines */}
      <div style={{ opacity: wiresFade, display: "flex", flexDirection: "column", gap: 8, marginTop: 30 }}>
        {Array.from({ length: 8 }, (_, i) => (
          <div key={i} style={{ height: 1, background: `linear-gradient(90deg, ${COLORS.accent}ee, ${COLORS.accent}14)` }} />
        ))}
      </div>

      <div
        style={{
          opacity: noteFade,
          fontFamily: sans,
          fontSize: 17,
          color: COLORS.textDim,
          marginTop: 16,
        }}
      >
        The quantum circuit doesn't replace the classical model {"\u2014"} it corrects it.
        This hybrid design converges faster and needs fewer training samples to reach peak accuracy.
      </div>
    </AbsoluteFill>
  );
};
