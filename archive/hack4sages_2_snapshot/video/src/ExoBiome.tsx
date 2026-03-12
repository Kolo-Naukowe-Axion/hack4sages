import React from "react";
import { AbsoluteFill } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { COLORS } from "./lib/constants";
import { TitleScene } from "./scenes/TitleScene";
import { ChallengeScene } from "./scenes/ChallengeScene";
import { TransmissionScene } from "./scenes/TransmissionScene";
import { SpectrumScene } from "./scenes/SpectrumScene";
import { RetrievalScene } from "./scenes/RetrievalScene";
import { ArchitectureScene } from "./scenes/ArchitectureScene";
import { PredictionsScene } from "./scenes/PredictionsScene";
import { SignificanceScene } from "./scenes/SignificanceScene";
import { ClosingScene } from "./scenes/ClosingScene";

// 30 fps, 120 seconds = 3600 frames
// 9 scenes, 8 fade transitions of 15 frames each = 120 frames eaten
// Raw scene durations sum to 3720, net = 3600
//
// [0-12s]    Title:         Can we detect life from starlight?
// [12-26s]   Challenge:     What is a biosignature? The 5 gases.
// [26-42s]   Transmission:  How transmission spectroscopy works.
// [42-58s]   Spectrum:      Animated observed vs retrieved chart.
// [58-72s]   Retrieval:     The inverse problem + dataset.
// [72-86s]   Architecture:  Hybrid quantum-classical approach.
// [86-100s]  Predictions:   Per-gas RMSE bars + overall metric.
// [100-112s] Significance:  Verified checkpoints + model lineup.
// [112-120s] Closing:       Vision + key numbers.

const FADE_DURATION = 15;

const SCENES = [
  { component: TitleScene, duration: 360 + FADE_DURATION },
  { component: ChallengeScene, duration: 420 + FADE_DURATION },
  { component: TransmissionScene, duration: 480 + FADE_DURATION },
  { component: SpectrumScene, duration: 480 + FADE_DURATION },
  { component: RetrievalScene, duration: 420 + FADE_DURATION },
  { component: ArchitectureScene, duration: 420 + FADE_DURATION },
  { component: PredictionsScene, duration: 420 + FADE_DURATION },
  { component: SignificanceScene, duration: 360 + FADE_DURATION },
  { component: ClosingScene, duration: 240 },
];

export const ExoBiome: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      <TransitionSeries>
        {SCENES.flatMap((scene, i) => {
          const Scene = scene.component;
          const elements: React.ReactNode[] = [];

          elements.push(
            <TransitionSeries.Sequence key={`scene-${i}`} durationInFrames={scene.duration}>
              <Scene />
            </TransitionSeries.Sequence>,
          );

          if (i < SCENES.length - 1) {
            elements.push(
              <TransitionSeries.Transition
                key={`transition-${i}`}
                presentation={fade()}
                timing={linearTiming({ durationInFrames: FADE_DURATION })}
              />,
            );
          }

          return elements;
        })}
      </TransitionSeries>
    </AbsoluteFill>
  );
};
