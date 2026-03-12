import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig, Easing } from "remotion";
import { sans, mono } from "../lib/fonts";
import { COLORS } from "../lib/constants";
import { performance, modelRoster } from "../lib/data";

export const SignificanceScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const sectionFade = interpolate(frame, [0, 0.8 * fps], [0, 1], { extrapolateRight: "clamp" });

  const headingFade = interpolate(frame, [0.3 * fps, 1.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const descFade = interpolate(frame, [1 * fps, 2.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const lineupFade = interpolate(frame, [6.5 * fps, 8.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const hardwareFade = interpolate(frame, [8.5 * fps, 10.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const ADVANTAGES = [
    { value: "6", label: "epochs to converge", note: "Reaches peak accuracy where classical models need significantly more training." },
    { value: "0.2994", label: "holdout mRMSE", note: "Competitive with the state-of-the-art on the same dataset." },
    { value: "Real HW", label: "IQM Spark 8-qubit", note: "Not a simulation \u2014 tested and validated on a real quantum processor." },
  ];

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
        07 / the quantum advantage
      </div>

      <div style={{ opacity: headingFade, fontFamily: sans, fontSize: 52, fontWeight: 800, lineHeight: 0.92, letterSpacing: -2, color: COLORS.text, textTransform: "uppercase", marginBottom: 16 }}>
        Faster training.
        <br />
        Less data needed.
        <br />
        Real hardware.
      </div>

      <div
        style={{
          opacity: descFade,
          fontFamily: sans,
          fontSize: 20,
          fontWeight: 400,
          color: COLORS.textMuted,
          lineHeight: 1.6,
          maxWidth: 800,
          marginBottom: 50,
        }}
      >
        The hybrid quantum model converges in fewer epochs than purely classical approaches
        and reaches competitive accuracy with the same limited training data {"\u2014"} validated on
        a real quantum computer, not just a simulator.
      </div>

      {/* Three advantage columns */}
      <div style={{ display: "flex", gap: 50 }}>
        {ADVANTAGES.map((adv, i) => {
          const delay = 2.5 * fps + i * 20;
          const colFade = interpolate(frame, [delay, delay + 1 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.quad),
          });

          return (
            <div key={adv.label} style={{ opacity: colFade, flex: 1 }}>
              <div style={{ fontFamily: sans, fontSize: 64, fontWeight: 800, lineHeight: 0.9, letterSpacing: -3, color: COLORS.accent }}>
                {adv.value}
              </div>
              <div style={{ fontFamily: mono, fontSize: 13, letterSpacing: 3, textTransform: "uppercase", color: COLORS.textDim, marginTop: 8, marginBottom: 8 }}>
                {adv.label}
              </div>
              <div style={{ fontFamily: sans, fontSize: 15, fontWeight: 400, color: COLORS.textMuted, lineHeight: 1.5 }}>
                {adv.note}
              </div>
            </div>
          );
        })}
      </div>

      {/* Model lineup */}
      <div
        style={{
          opacity: lineupFade,
          display: "flex",
          gap: 30,
          marginTop: 45,
          paddingTop: 20,
          borderTop: `1px solid ${COLORS.border}`,
        }}
      >
        {modelRoster.map((model) => {
          const isOurs = model.position === "team highlight";
          return (
            <div
              key={model.name}
              style={{
                padding: "14px 20px",
                border: isOurs ? `2px solid ${COLORS.accent}` : `1px solid ${COLORS.border}`,
                borderRadius: 6,
                flex: 1,
              }}
            >
              <div style={{ fontFamily: sans, fontSize: 16, fontWeight: 700, color: isOurs ? COLORS.accent : COLORS.text, textTransform: "uppercase" }}>
                {model.name}
              </div>
              <div style={{ fontFamily: mono, fontSize: 12, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 2, marginTop: 4 }}>
                {model.className} &middot; {model.position}
              </div>
            </div>
          );
        })}
      </div>

      {/* Real hardware callout */}
      <div
        style={{
          opacity: hardwareFade,
          fontFamily: sans,
          fontSize: 18,
          fontWeight: 600,
          color: COLORS.accent,
          marginTop: 20,
        }}
      >
        Executed on IQM Spark at PWR Wroc{"\u0142"}aw {"\u2014"} real quantum advantage, not theoretical.
      </div>
    </AbsoluteFill>
  );
};
