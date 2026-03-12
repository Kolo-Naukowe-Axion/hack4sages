import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig, Easing } from "remotion";
import { sans, mono } from "../lib/fonts";
import { COLORS } from "../lib/constants";
import { performance } from "../lib/data";

const maxValue = Math.max(...performance.perGasRmse.map((e) => e.value));

export const PredictionsScene: React.FC = () => {
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

  // Big number
  const bigFade = interpolate(frame, [7 * fps, 9 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
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
        06 / how well do we recover the chemistry
      </div>

      <div style={{ opacity: headingFade, fontFamily: sans, fontSize: 60, fontWeight: 800, lineHeight: 0.92, letterSpacing: -3, color: COLORS.text, textTransform: "uppercase", maxWidth: 800, marginBottom: 16 }}>
        Per-gas retrieval
        <br />
        accuracy.
      </div>

      <div
        style={{
          opacity: descFade,
          fontFamily: sans,
          fontSize: 19,
          fontWeight: 400,
          color: COLORS.textMuted,
          lineHeight: 1.5,
          maxWidth: 800,
          marginBottom: 45,
        }}
      >
        Root mean squared error between predicted and true log{"\u2081\u2080"} volume mixing ratio.
        Lower means the model recovers the real abundance more faithfully.
      </div>

      {/* Bar chart */}
      <div style={{ display: "flex", flexDirection: "column", gap: 24, maxWidth: 1400 }}>
        {performance.perGasRmse.map((entry, i) => {
          const delay = 2.5 * fps + i * 10;
          const barSpring = spring({ frame: frame - delay, fps, config: { damping: 200 } });
          const valueFade = interpolate(frame, [delay + 0.8 * fps, delay + 1.3 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const barWidth = (entry.value / maxValue) * 100 * barSpring;

          return (
            <div key={entry.gas} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <span style={{ fontFamily: sans, fontSize: 28, fontWeight: 800, color: COLORS.accent, letterSpacing: -1 }}>
                  {entry.gas}
                </span>
                <span style={{ fontFamily: mono, fontSize: 20, fontWeight: 500, color: COLORS.text, opacity: valueFade }}>
                  {entry.value.toFixed(4)}
                </span>
              </div>
              <div style={{ height: 14, borderRadius: 999, backgroundColor: COLORS.gridLine, overflow: "hidden" }}>
                <div
                  style={{
                    width: `${barWidth}%`,
                    height: "100%",
                    borderRadius: 999,
                    background: `linear-gradient(90deg, ${COLORS.barGradientStart}, ${COLORS.barGradientEnd})`,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Overall metric */}
      <div
        style={{
          opacity: bigFade,
          display: "flex",
          alignItems: "baseline",
          gap: 20,
          marginTop: 45,
          paddingTop: 24,
          borderTop: `1px solid ${COLORS.border}`,
        }}
      >
        <span style={{ fontFamily: mono, fontSize: 14, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 3 }}>
          mean across all 5 gases (holdout)
        </span>
        <span style={{ fontFamily: sans, fontSize: 52, fontWeight: 800, color: COLORS.accent, letterSpacing: -3 }}>
          {performance.holdout.toFixed(4)}
        </span>
        <span style={{ fontFamily: sans, fontSize: 22, fontWeight: 400, color: COLORS.textMuted }}>
          mRMSE
        </span>
      </div>
    </AbsoluteFill>
  );
};
