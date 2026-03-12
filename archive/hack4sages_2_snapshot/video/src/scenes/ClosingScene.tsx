import React, { useMemo } from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig, Easing, random } from "remotion";
import { sans, mono } from "../lib/fonts";
import { COLORS } from "../lib/constants";

export const ClosingScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const stars = useMemo(
    () =>
      Array.from({ length: 80 }, (_, i) => ({
        x: random(`close-x-${i}`) * 100,
        y: random(`close-y-${i}`) * 100,
        size: 0.3 + random(`close-s-${i}`) * 1.5,
        baseOpacity: 0.15 + random(`close-o-${i}`) * 0.35,
        speed: 3 + random(`close-d-${i}`) * 5,
      })),
    [],
  );

  const headingFade = interpolate(frame, [0, 1.8 * fps], [0, 1], {
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });
  const headingSlide = interpolate(frame, [0, 1.8 * fps], [30, 0], {
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const subFade = interpolate(frame, [1.5 * fps, 3 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const duration = 8 * fps;
  const fadeOut = interpolate(frame, [duration - 2 * fps, duration], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const tagFade = interpolate(frame, [3.5 * fps, 5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const FACTS = [
    { value: "6", label: "epochs to converge" },
    { value: "0.2994", label: "holdout mRMSE" },
    { value: "8", label: "qubits \u00b7 real IQM HW" },
    { value: "5", label: "gases jointly" },
  ];

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(circle at 18% 12%, rgba(255,120,36,0.12), transparent 24rem), linear-gradient(180deg, #090a0d 0%, #0b0c10 48%, #07080b 100%)`,
        padding: "80px 120px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        opacity: fadeOut,
        overflow: "hidden",
      }}
    >
      {stars.map((star, i) => {
        const twinkle = star.baseOpacity * (0.5 + 0.5 * Math.sin((frame / fps) * ((2 * Math.PI) / star.speed) + i));
        return (
          <div
            key={i}
            style={{ position: "absolute", left: `${star.x}%`, top: `${star.y}%`, width: star.size, height: star.size, borderRadius: "50%", backgroundColor: COLORS.text, opacity: twinkle }}
          />
        );
      })}

      <div
        style={{
          opacity: headingFade,
          transform: `translateY(${headingSlide}px)`,
          fontFamily: sans,
          fontSize: 76,
          fontWeight: 800,
          lineHeight: 0.92,
          letterSpacing: -3,
          color: COLORS.text,
          textTransform: "uppercase",
          maxWidth: 1100,
          zIndex: 1,
        }}
      >
        Less data.
        <br />
        Faster training.
        <br />
        Real quantum hardware.
      </div>

      <div
        style={{
          opacity: subFade,
          fontFamily: sans,
          fontSize: 22,
          fontWeight: 400,
          color: COLORS.textMuted,
          lineHeight: 1.6,
          maxWidth: 700,
          marginTop: 30,
          zIndex: 1,
        }}
      >
        ExoBiome shows that a hybrid quantum-classical model can recover
        atmospheric abundances faster and with less data than purely classical approaches
        {"\u2014"} validated on real quantum hardware.
      </div>

      {/* Key facts */}
      <div style={{ display: "flex", gap: 60, marginTop: 50, zIndex: 1 }}>
        {FACTS.map((fact, i) => {
          const delay = 2 * fps + i * 8;
          const factFade = interpolate(frame, [delay, delay + 0.8 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.quad),
          });

          return (
            <div key={fact.label} style={{ opacity: factFade, textAlign: "center" }}>
              <div style={{ fontFamily: sans, fontSize: 44, fontWeight: 800, color: COLORS.accent, lineHeight: 1 }}>
                {fact.value}
              </div>
              <div style={{ fontFamily: mono, fontSize: 13, color: COLORS.textDim, marginTop: 6, textTransform: "uppercase", letterSpacing: 2 }}>
                {fact.label}
              </div>
            </div>
          );
        })}
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 50,
          left: 120,
          fontFamily: mono,
          fontSize: 14,
          letterSpacing: 4,
          textTransform: "uppercase",
          color: COLORS.textDim,
          opacity: tagFade * fadeOut,
          zIndex: 1,
        }}
      >
        ExoBiome &bull; HACK-4-SAGES 2026 &bull; ETH Zurich
      </div>
    </AbsoluteFill>
  );
};
