import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig, Easing } from "remotion";
import { sans, mono } from "../lib/fonts";
import { COLORS } from "../lib/constants";

export const TransmissionScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const sectionFade = interpolate(frame, [0, 0.8 * fps], [0, 1], { extrapolateRight: "clamp" });

  const headingFade = interpolate(frame, [0.3 * fps, 1.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const diagramFade = interpolate(frame, [2 * fps, 3.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  // Star
  const starFade = interpolate(frame, [3 * fps, 4 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Light rays progress
  const raysProgress = interpolate(frame, [4 * fps, 7 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  // Planet
  const planetFade = interpolate(frame, [5 * fps, 6 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Atmosphere glow
  const atmoFade = interpolate(frame, [6 * fps, 8 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Telescope
  const telFade = interpolate(frame, [7 * fps, 8.5 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Labels
  const labelsFade = interpolate(frame, [8 * fps, 10 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Caption
  const captionFade = interpolate(frame, [10 * fps, 12 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const starX = 260;
  const planetX = 960;
  const telX = 1660;
  const centerY = 540;

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
        02 / transmission spectroscopy
      </div>

      <div
        style={{
          position: "absolute",
          top: 80,
          left: 120,
          opacity: headingFade,
          fontFamily: sans,
          fontSize: 44,
          fontWeight: 800,
          letterSpacing: -2,
          color: COLORS.text,
          textTransform: "uppercase",
        }}
      >
        Starlight passes through an atmosphere. We read the gaps.
      </div>

      <svg width={1920} height={1080} style={{ position: "absolute" }}>
        {/* Star */}
        <g opacity={starFade}>
          <circle cx={starX} cy={centerY} r={70} fill="#fbbf24" />
          <circle cx={starX} cy={centerY} r={90} fill="none" stroke="#fbbf2444" strokeWidth={2} />
          <circle cx={starX} cy={centerY} r={110} fill="none" stroke="#fbbf2422" strokeWidth={1} />
        </g>

        {/* Light rays (from star to telescope, passing through planet atmosphere) */}
        {[-50, -25, 0, 25, 50].map((offset, i) => {
          const rayLength = (telX - starX - 140) * raysProgress;
          return (
            <line
              key={i}
              x1={starX + 80}
              y1={centerY + offset * 0.6}
              x2={starX + 80 + rayLength}
              y2={centerY + offset}
              stroke="#fbbf2488"
              strokeWidth={1.5}
              strokeDasharray="8 6"
            />
          );
        })}

        {/* Planet */}
        <g opacity={planetFade}>
          <circle cx={planetX} cy={centerY} r={55} fill="#1e293b" />
          <circle cx={planetX} cy={centerY} r={55} fill="none" stroke="#334155" strokeWidth={2} />
        </g>

        {/* Atmosphere ring */}
        <g opacity={atmoFade}>
          <circle cx={planetX} cy={centerY} r={78} fill="none" stroke={COLORS.accent} strokeWidth={3} strokeDasharray="6 4" strokeOpacity={0.6} />
          <circle cx={planetX} cy={centerY} r={88} fill="none" stroke={COLORS.accent} strokeWidth={1} strokeOpacity={0.2} />
        </g>

        {/* Telescope */}
        <g opacity={telFade}>
          <rect x={telX - 30} y={centerY - 20} width={60} height={40} rx={6} fill="none" stroke={COLORS.text} strokeWidth={2} />
          <line x1={telX - 30} y1={centerY} x2={telX - 55} y2={centerY - 15} stroke={COLORS.text} strokeWidth={2} />
          <line x1={telX - 30} y1={centerY} x2={telX - 55} y2={centerY + 15} stroke={COLORS.text} strokeWidth={2} />
        </g>

        {/* Labels */}
        <g opacity={labelsFade}>
          <text x={starX} y={centerY + 130} fill={COLORS.textDim} fontSize={16} textAnchor="middle" fontFamily={mono}>
            Host star
          </text>
          <text x={planetX} y={centerY + 130} fill={COLORS.accent} fontSize={16} textAnchor="middle" fontFamily={mono}>
            Exoplanet + atmosphere
          </text>
          <text x={telX} y={centerY + 130} fill={COLORS.textDim} fontSize={16} textAnchor="middle" fontFamily={mono}>
            Telescope (Ariel)
          </text>
        </g>

        {/* Absorption annotation */}
        <g opacity={labelsFade}>
          <text x={planetX} y={centerY - 110} fill={COLORS.accent} fontSize={18} textAnchor="middle" fontFamily={sans} fontWeight={600}>
            Molecules absorb specific wavelengths
          </text>
          <line x1={planetX} y1={centerY - 95} x2={planetX} y2={centerY - 80} stroke={COLORS.accent} strokeWidth={1.5} strokeOpacity={0.5} />
        </g>
      </svg>

      {/* Bottom caption */}
      <div
        style={{
          position: "absolute",
          bottom: 70,
          left: 120,
          right: 120,
          opacity: captionFade,
          fontFamily: sans,
          fontSize: 22,
          fontWeight: 400,
          color: COLORS.textMuted,
          lineHeight: 1.6,
          maxWidth: 900,
        }}
      >
        Each gas leaves a unique absorption fingerprint. By measuring which wavelengths
        are missing, we can determine what molecules are present \u2014 and in what abundance.
      </div>
    </AbsoluteFill>
  );
};
