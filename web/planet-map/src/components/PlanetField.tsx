"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Planet } from "@/types";

function tempToColor(temp: number | null): string {
  if (temp === null) return "#3b82f6";
  if (temp > 350) return "#ef4444";
  if (temp > 250) return "#f59e0b";
  if (temp > 200) return "#22c55e";
  return "#3b82f6";
}

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

interface PlanetNode {
  planet: Planet;
  x: number;
  y: number;
  radius: number;
  baseX: number;
  baseY: number;
}

interface Props {
  planets: Planet[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  compact?: boolean;
}

export function PlanetField({ planets, selectedId, onSelect, compact = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<PlanetNode[]>([]);
  const hoveredRef = useRef<string | null>(null);
  const [hovered, setHovered] = useState<{ planet: Planet; x: number; y: number } | null>(null);
  const dprRef = useRef(1);

  const buildNodes = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const w = canvas.width;
    const h = canvas.height;
    const dpr = dprRef.current;
    const padL = compact ? 40 : 80 * dpr;
    const padR = compact ? 40 : 60 * dpr;
    const padT = compact ? 40 : 50 * dpr;
    const padB = compact ? 40 : 90 * dpr;

    const sorted = [...planets];
    sorted.sort((a, b) => a.distanceLy - b.distanceLy);
    const distRank = new Map(sorted.map((p, i) => [p.id, planets.length > 1 ? i / (planets.length - 1) : 0.5]));

    sorted.sort((a, b) => (a.eqTempK ?? 250) - (b.eqTempK ?? 250));
    const tempRank = new Map(sorted.map((p, i) => [p.id, planets.length > 1 ? i / (planets.length - 1) : 0.5]));

    const temps = planets.map((p) => p.eqTempK ?? 250);
    const dists = planets.map((p) => p.distanceLy);
    const minTemp = Math.min(...temps);
    const maxTemp = Math.max(...temps);
    const minDist = Math.min(...dists);
    const maxDist = Math.max(...dists);

    const radiuses = planets.map((p) => p.radiusEarth);
    const minR = Math.min(...radiuses);
    const maxR = Math.max(...radiuses);

    const blend = 0.55;

    const nodes: PlanetNode[] = planets.map((planet) => {
      const temp = planet.eqTempK ?? 250;
      const valueDist =
        maxDist === minDist
          ? 0.5
          : Math.log(planet.distanceLy - minDist + 1) / Math.log(maxDist - minDist + 1);
      const valueTemp = maxTemp === minTemp ? 0.5 : (temp - minTemp) / (maxTemp - minTemp);

      const normDist = blend * (distRank.get(planet.id) ?? 0.5) + (1 - blend) * valueDist;
      const normTemp = blend * (tempRank.get(planet.id) ?? 0.5) + (1 - blend) * valueTemp;

      const x = padL + normDist * (w - padL - padR);
      const y = padT + (1 - normTemp) * (h - padT - padB);

      const normRadius = maxR === minR ? 0.5 : (planet.radiusEarth - minR) / (maxR - minR);
      const minPx = (compact ? 6 : 8) * dpr;
      const maxPx = (compact ? 18 : 28) * dpr;
      const radius = minPx + normRadius * (maxPx - minPx);

      return { planet, x, y, radius, baseX: x, baseY: y };
    });

    resolveOverlaps(nodes, w, h, padL, padR, padT, padB);
    nodesRef.current = nodes;
  }, [planets, compact]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const dpr = dprRef.current;
    const padL = compact ? 40 : 80 * dpr;
    const padR = compact ? 40 : 60 * dpr;
    const padT = compact ? 40 : 50 * dpr;
    const padB = compact ? 40 : 90 * dpr;

    ctx.clearRect(0, 0, w, h);

    const rng = seededRandom(42);
    const starCount = Math.floor((w * h) / 12000);
    for (let i = 0; i < starCount; i++) {
      const sx = rng() * w;
      const sy = rng() * h;
      const alpha = 0.08 + rng() * 0.18;
      const size = (0.5 + rng() * 0.8) * dpr;
      ctx.beginPath();
      ctx.arc(sx, sy, size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
      ctx.fill();
    }

    const gridColor = "rgba(42, 45, 55, 0.6)";
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    const gridCountX = compact ? 4 : 6;
    const gridCountY = compact ? 3 : 5;

    for (let i = 0; i <= gridCountX; i++) {
      const x = padL + (i / gridCountX) * (w - padL - padR);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
    }
    for (let i = 0; i <= gridCountY; i++) {
      const y = padT + (i / gridCountY) * (h - padT - padB);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
    }

    if (!compact) {
      ctx.fillStyle = "#9ca3af";
      ctx.font = `${Math.round(11 * dpr)}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText("Distance (ly) \u2192", (padL + w - padR) / 2, h - padB + 30 * dpr);

      ctx.save();
      ctx.translate(20 * dpr, (padT + h - padB) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Temperature (K) \u2192", 0, 0);
      ctx.restore();
    }

    const nodes = nodesRef.current;
    for (const node of nodes) {
      const isHovered = hoveredRef.current === node.planet.id;
      const isSelected = selectedId === node.planet.id;
      const color = tempToColor(node.planet.eqTempK);

      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      if (isSelected) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius + 3 * dpr, 0, Math.PI * 2);
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 2 * dpr;
        ctx.stroke();
      }

      if (isHovered && !isSelected) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius + 3 * dpr, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
        ctx.lineWidth = 2 * dpr;
        ctx.stroke();
      }

      if (node.planet.inHabitableZone) {
        const dotX = node.x + node.radius * 0.7;
        const dotY = node.y - node.radius * 0.7;
        ctx.beginPath();
        ctx.arc(dotX, dotY, 3 * dpr, 0, Math.PI * 2);
        ctx.fillStyle = "#22c55e";
        ctx.fill();
      }

      if (node.planet.hasJWSTData) {
        const bx = node.x + node.radius * 0.7;
        const by = node.planet.inHabitableZone
          ? node.y - node.radius * 0.7 + 8 * dpr
          : node.y - node.radius * 0.7;
        ctx.fillStyle = "#9ca3af";
        ctx.font = `bold ${Math.round(8 * dpr)}px "JetBrains Mono", monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("J", bx, by);
      }

      if (!compact || isHovered || isSelected) {
        ctx.fillStyle = isSelected
          ? "#3b82f6"
          : isHovered
          ? "#e5e7eb"
          : "#9ca3af";
        ctx.font = `${isSelected || isHovered ? "500" : "400"} ${Math.round(
          (compact ? 9 : 11) * dpr
        )}px Inter, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(node.planet.name, node.x, node.y + node.radius + 4 * dpr);
      }
    }
  }, [compact, selectedId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    function resize() {
      if (!canvas) return;
      dprRef.current = window.devicePixelRatio || 1;
      canvas.width = canvas.offsetWidth * dprRef.current;
      canvas.height = canvas.offsetHeight * dprRef.current;
      buildNodes();
      draw();
    }

    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [buildNodes, draw]);

  useEffect(() => {
    draw();
  }, [draw, hovered]);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * dprRef.current;
      const my = (e.clientY - rect.top) * dprRef.current;

      let found: PlanetNode | null = null;
      for (const node of nodesRef.current) {
        const dx = mx - node.x;
        const dy = my - node.y;
        if (dx * dx + dy * dy < (node.radius + 8 * dprRef.current) ** 2) {
          found = node;
          break;
        }
      }

      const newId = found?.planet.id ?? null;
      if (newId !== hoveredRef.current) {
        hoveredRef.current = newId;
        if (found) {
          setHovered({
            planet: found.planet,
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
          });
        } else {
          setHovered(null);
        }
      } else if (found) {
        setHovered({
          planet: found.planet,
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
        });
      }

      canvas.style.cursor = found ? "pointer" : "default";
    },
    []
  );

  const handleClick = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * dprRef.current;
      const my = (e.clientY - rect.top) * dprRef.current;

      for (const node of nodesRef.current) {
        const dx = mx - node.x;
        const dy = my - node.y;
        if (dx * dx + dy * dy < (node.radius + 8 * dprRef.current) ** 2) {
          onSelect(node.planet.id);
          return;
        }
      }
    },
    [onSelect]
  );

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        onPointerMove={handlePointerMove}
        onPointerDown={handleClick}
        onPointerLeave={() => {
          hoveredRef.current = null;
          setHovered(null);
        }}
      />

      {hovered && (
        <div
          className="absolute z-30 pointer-events-none fade-up"
          style={{
            left: hovered.x + 16,
            top: hovered.y - 10,
          }}
        >
          <div className="bg-surface border border-border rounded-lg px-3 py-2.5 min-w-[170px]">
            <div className="flex items-center gap-2 mb-1.5">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ background: tempToColor(hovered.planet.eqTempK) }}
              />
              <span className="font-medium text-text text-sm">
                {hovered.planet.name}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs">
              <span className="text-muted">Temp</span>
              <span className="text-text font-mono">
                {hovered.planet.eqTempK ?? "N/A"} K
              </span>
              <span className="text-muted">Distance</span>
              <span className="text-text font-mono">
                {hovered.planet.distanceLy} ly
              </span>
              <span className="text-muted">Radius</span>
              <span className="text-text font-mono">
                {hovered.planet.radiusEarth} R&#8853;
              </span>
              {hovered.planet.inHabitableZone && (
                <span className="col-span-2 text-accent-green mt-1 text-[11px] font-medium">
                  Habitable Zone
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function resolveOverlaps(
  nodes: PlanetNode[],
  w: number,
  h: number,
  padL: number,
  padR: number,
  padT: number,
  padB: number,
) {
  for (let iter = 0; iter < 30; iter++) {
    let moved = false;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        const dx = b.baseX - a.baseX;
        const dy = b.baseY - a.baseY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const minDist = a.radius + b.radius + 18;

        if (dist < minDist) {
          const overlap = minDist - dist;
          const angle =
            dist === 0 ? Math.random() * Math.PI * 2 : Math.atan2(dy, dx);
          const push = overlap * 0.55;

          a.baseX -= Math.cos(angle) * push;
          a.baseY -= Math.sin(angle) * push;
          b.baseX += Math.cos(angle) * push;
          b.baseY += Math.sin(angle) * push;

          a.baseX = Math.max(padL + a.radius, Math.min(w - padR - a.radius, a.baseX));
          a.baseY = Math.max(padT + a.radius, Math.min(h - padB - a.radius, a.baseY));
          b.baseX = Math.max(padL + b.radius, Math.min(w - padR - b.radius, b.baseX));
          b.baseY = Math.max(padT + b.radius, Math.min(h - padB - b.radius, b.baseY));

          moved = true;
        }
      }
    }
    if (!moved) break;
  }

  for (const n of nodes) {
    n.x = n.baseX;
    n.y = n.baseY;
  }
}
