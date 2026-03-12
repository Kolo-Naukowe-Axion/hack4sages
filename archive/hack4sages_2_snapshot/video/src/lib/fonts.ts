import { loadFont as loadJetBrains } from "@remotion/google-fonts/JetBrainsMono";
import { loadFont as loadInter } from "@remotion/google-fonts/Inter";

const { fontFamily: mono } = loadJetBrains("normal", {
  weights: ["300", "400", "500", "700"],
  subsets: ["latin"],
});

const { fontFamily: sans } = loadInter("normal", {
  weights: ["300", "400", "500", "600", "700", "800"],
  subsets: ["latin"],
});

export { mono, sans };
