import { Composition } from "remotion";
import { ExoBiome } from "./ExoBiome";
import { FPS, WIDTH, HEIGHT, DURATION_FRAMES } from "./lib/constants";

export const RemotionRoot = () => {
  return (
    <Composition
      id="ExoBiome"
      component={ExoBiome}
      durationInFrames={DURATION_FRAMES}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
    />
  );
};
