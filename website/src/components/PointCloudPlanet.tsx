function buildPoints() {
  const points: Array<{ x: number; y: number; size: number; opacity: number }> = [];

  for (let row = 0; row < 92; row += 1) {
    for (let col = 0; col < 92; col += 1) {
      const x = col / 91;
      const y = row / 91;
      const dx = x - 0.5;
      const dy = y - 0.66;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const haloDistance = Math.sqrt((x - 0.5) ** 2 + (y - 0.54) ** 2);
      const shell = distance < 0.35;
      const halo = haloDistance > 0.28 && haloDistance < 0.43 && y < 0.7;
      const band = Math.abs(y - 0.59) < 0.055;
      const pseudo = (row * 37 + col * 19 + row * col) % 17;

      if ((shell && pseudo < 6) || (halo && pseudo < 4) || (band && pseudo < 2)) {
        points.push({
          x: 40 + x * 720,
          y: 20 + y * 560,
          size: pseudo % 3 === 0 ? 5 : 4,
          opacity: shell ? 0.72 : 0.48,
        });
      }
    }
  }

  return points;
}

const points = buildPoints();

export function PointCloudPlanet() {
  return (
    <div className="planet-figure" aria-hidden="true">
      <svg viewBox="0 0 800 620" className="planet-svg">
        {points.map((point, index) => (
          <rect
            key={index}
            x={point.x}
            y={point.y}
            width={point.size}
            height={point.size}
            fill="rgba(18, 16, 13, 0.72)"
            opacity={point.opacity}
          />
        ))}
      </svg>
    </div>
  );
}
