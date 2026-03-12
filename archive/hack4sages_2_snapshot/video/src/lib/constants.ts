export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;
export const DURATION_FRAMES = FPS * 120; // 3600 frames

// Colors from web presentation-lab (Monolith Keynote / variant 6)
export const COLORS = {
  bg: "#090a0d",
  text: "#f3efe7",
  textMuted: "rgba(243, 239, 231, 0.78)",
  textDim: "rgba(243, 239, 231, 0.56)",
  textFaint: "rgba(243, 239, 231, 0.18)",
  accent: "#ff8f4d",       // orange accent from web
  gridLine: "rgba(255, 255, 255, 0.08)",
  border: "rgba(255, 255, 255, 0.08)",

  // Spectrum lines
  observed: "rgba(243, 239, 231, 0.88)",
  retrieved: "#ff8f4d",

  // Per-gas bar gradient
  barGradientStart: "#0d5f63",
  barGradientEnd: "#d79a2b",

  // Blueprint accent (for architecture)
  cyan: "#79eeff",
  cyanMuted: "rgba(120, 239, 255, 0.48)",
  blueprintBg: "rgba(2, 28, 38, 0.86)",
} as const;
