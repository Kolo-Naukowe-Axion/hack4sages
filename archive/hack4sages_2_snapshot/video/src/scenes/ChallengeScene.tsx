import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig, Easing } from "remotion";
import { sans, mono } from "../lib/fonts";
import { COLORS } from "../lib/constants";

const GASES = [
  { key: "H\u2082O", label: "Water", why: "Sets the stage \u2014 stable volatile, habitability prerequisite." },
  { key: "CO\u2082", label: "Carbon Dioxide", why: "Carbon-cycle context, greenhouse reference gas." },
  { key: "CO", label: "Carbon Monoxide", why: "Counterweight \u2014 rejects false-positive methane stories." },
  { key: "CH\u2084", label: "Methane", why: "Classic biosignature candidate \u2014 compelling with CO\u2082 context." },
  { key: "NH\u2083", label: "Ammonia", why: "Trace gas that pushes beyond the obvious, research-grade signal." },
];

export const ChallengeScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const sectionFade = interpolate(frame, [0, 0.8 * fps], [0, 1], { extrapolateRight: "clamp" });

  const headingFade = interpolate(frame, [0.3 * fps, 1.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const contextFade = interpolate(frame, [1 * fps, 2.5 * fps], [0, 1], {
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
        01 / what is a biosignature
      </div>

      <div style={{ opacity: headingFade, fontFamily: sans, fontSize: 64, fontWeight: 800, lineHeight: 0.92, letterSpacing: -3, color: COLORS.text, textTransform: "uppercase", maxWidth: 900, marginBottom: 20 }}>
        No single gas
        <br />
        proves life.
      </div>

      <div
        style={{
          opacity: contextFade,
          fontFamily: sans,
          fontSize: 22,
          fontWeight: 400,
          color: COLORS.textMuted,
          lineHeight: 1.6,
          maxWidth: 800,
          marginBottom: 45,
        }}
      >
        A biosignature is a chemical disequilibrium in an atmosphere \u2014
        gases that shouldn&apos;t coexist unless something is producing them.
        The signal is in their <em style={{ color: COLORS.accent, fontStyle: "normal", fontWeight: 700 }}>co-presence</em>, not any single molecule.
      </div>

      {/* Gas rows */}
      <div style={{ display: "flex", flexDirection: "column" }}>
        {GASES.map((gas, i) => {
          const delay = 2.5 * fps + i * 10;
          const rowFade = interpolate(frame, [delay, delay + 0.6 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.quad),
          });

          return (
            <div
              key={gas.key}
              style={{
                opacity: rowFade,
                display: "flex",
                alignItems: "flex-end",
                gap: 36,
                padding: "16px 0",
                borderBottom: `1px solid ${COLORS.border}`,
              }}
            >
              <div style={{ fontFamily: sans, fontSize: 50, fontWeight: 800, lineHeight: 0.9, letterSpacing: -2, color: COLORS.accent, minWidth: 160 }}>
                {gas.key}
              </div>
              <div>
                <div style={{ fontFamily: sans, fontSize: 20, fontWeight: 700, textTransform: "uppercase", color: COLORS.text, marginBottom: 2 }}>
                  {gas.label}
                </div>
                <div style={{ fontFamily: sans, fontSize: 16, fontWeight: 400, color: COLORS.textMuted, lineHeight: 1.5 }}>
                  {gas.why}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
