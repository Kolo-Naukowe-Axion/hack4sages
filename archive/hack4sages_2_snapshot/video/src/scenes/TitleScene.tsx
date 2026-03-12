import React, { useMemo } from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig, Easing, random } from "remotion";
import { sans, mono } from "../lib/fonts";
import { COLORS } from "../lib/constants";

export const TitleScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const stars = useMemo(
    () =>
      Array.from({ length: 140 }, (_, i) => ({
        x: random(`star-x-${i}`) * 100,
        y: random(`star-y-${i}`) * 100,
        size: 0.3 + random(`star-s-${i}`) * 2,
        baseOpacity: 0.2 + random(`star-o-${i}`) * 0.6,
        speed: 3 + random(`star-d-${i}`) * 6,
      })),
    [],
  );

  const headlineFade = interpolate(frame, [0, 2 * fps], [0, 1], {
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });
  const headlineSlide = interpolate(frame, [0, 2 * fps], [50, 0], {
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const subFade = interpolate(frame, [1.5 * fps, 3 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const tagFade = interpolate(frame, [2.5 * fps, 4 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(circle at 18% 12%, rgba(255,120,36,0.18), transparent 24rem), linear-gradient(180deg, #090a0d 0%, #0b0c10 48%, #07080b 100%)`,
        display: "flex",
        overflow: "hidden",
      }}
    >
      {stars.map((star, i) => {
        const twinkle = star.baseOpacity * (0.5 + 0.5 * Math.sin((frame / fps) * ((2 * Math.PI) / star.speed) + i));
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: `${star.x}%`,
              top: `${star.y}%`,
              width: star.size,
              height: star.size,
              borderRadius: "50%",
              backgroundColor: COLORS.text,
              opacity: twinkle,
            }}
          />
        );
      })}

      <div
        style={{
          position: "absolute",
          left: 120,
          bottom: 140,
          right: 120,
        }}
      >
        <div
          style={{
            opacity: headlineFade,
            transform: `translateY(${headlineSlide}px)`,
          }}
        >
          <div
            style={{
              fontFamily: mono,
              fontSize: 16,
              letterSpacing: 4,
              textTransform: "uppercase",
              color: COLORS.accent,
              marginBottom: 24,
            }}
          >
            Life Detection &amp; Biosignatures
          </div>
          <div
            style={{
              fontFamily: sans,
              fontSize: 120,
              fontWeight: 800,
              lineHeight: 0.88,
              letterSpacing: -5,
              color: COLORS.text,
              textTransform: "uppercase",
            }}
          >
            Can we detect
            <br />
            life from
            <br />
            starlight?
          </div>
        </div>

        <div
          style={{
            opacity: subFade,
            fontFamily: sans,
            fontSize: 24,
            fontWeight: 400,
            color: COLORS.textMuted,
            marginTop: 36,
            maxWidth: 700,
            lineHeight: 1.6,
          }}
        >
          ExoBiome reads the chemical fingerprints in exoplanet
          atmospheres to search for signs of biology.
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          top: 50,
          left: 120,
          opacity: tagFade,
          fontFamily: mono,
          fontSize: 14,
          letterSpacing: 4,
          textTransform: "uppercase",
          color: COLORS.textDim,
        }}
      >
        ExoBiome &bull; HACK-4-SAGES 2026 &bull; ETH Zurich
      </div>
    </AbsoluteFill>
  );
};
