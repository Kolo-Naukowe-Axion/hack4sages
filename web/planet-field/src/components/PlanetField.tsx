"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Planet } from "@/types";
import { Satellite } from "lucide-react";

function tempToColor(temp: number | null): string {
  if (temp === null) return "#8888cc";
  if (temp > 450) return "#ff6644";
  if (temp > 350) return "#ff9944";
  if (temp > 280) return "#44dd88";
  if (temp > 220) return "#44aaff";
  if (temp > 180) return "#6688ee";
  return "#9966dd";
}

function tempToGlow(temp: number | null): string {
  if (temp === null) return "rgba(136,136,204,0.4)";
  if (temp > 450) return "rgba(255,102,68,0.5)";
  if (temp > 350) return "rgba(255,153,68,0.5)";
  if (temp > 280) return "rgba(68,221,136,0.5)";
  if (temp > 220) return "rgba(68,170,255,0.5)";
  if (temp > 180) return "rgba(102,136,238,0.5)";
  return "rgba(153,102,221,0.5)";
}

interface PlanetNode {
  planet: Planet;
  x: number;
  y: number;
  radius: number;
  baseX: number;
  baseY: number;
  phase: number;
  speed: number;
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
  const animRef = useRef<number>(0);
  const dprRef = useRef(1);

  const buildNodes = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const w = canvas.width;
    const h = canvas.height;
    const padding = compact ? 40 : 60;

    const temps = planets.map((p) => p.eqTempK ?? 250);
    const dists = planets.map((p) => p.distanceLy);
    const minTemp = Math.min(...temps);
    const maxTemp = Math.max(...temps);
    const minDist = Math.min(...dists);
    const maxDist = Math.max(...dists);

    const nodes: PlanetNode[] = planets.map((planet, i) => {
      const temp = planet.eqTempK ?? 250;
      const normTemp = maxTemp === minTemp ? 0.5 : (temp - minTemp) / (maxTemp - minTemp);
      const normDist =
        maxDist === minDist ? 0.5 : Math.log(planet.distanceLy - minDist + 1) / Math.log(maxDist - minDist + 1);

      const x = padding + normDist * (w - padding * 2);
      const y = padding + (1 - normTemp) * (h - padding * 2);

      const baseRadius = compact ? 8 : 12;
      const sizeScale = compact ? 3 : 5;
      const radius = (baseRadius + planet.radiusEarth * sizeScale) * dprRef.current;

      return {
        planet,
        x,
        y,
        radius,
        baseX: x,
        baseY: y,
        phase: (i * 1.3 + planet.distanceLy * 0.1) % (Math.PI * 2),
        speed: 0.3 + (i % 5) * 0.15,
      };
    });

    resolveOverlaps(nodes, w, h, padding);
    nodesRef.current = nodes;
  }, [planets, compact]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    function resize() {
      if (!canvas) return;
      dprRef.current = window.devicePixelRatio || 1;
      canvas.width = canvas.offsetWidth * dprRef.current;
      canvas.height = canvas.offsetHeight * dprRef.current;
      buildNodes();
    }

    function draw(time: number) {
      if (!canvas || !ctx) return;
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const nodes = nodesRef.current;
      for (const node of nodes) {
        const t = time * 0.001;
        node.x = node.baseX + Math.sin(t * node.speed + node.phase) * 4 * dprRef.current;
        node.y = node.baseY + Math.cos(t * node.speed * 0.7 + node.phase) * 3 * dprRef.current;

        const isHovered = hoveredRef.current === node.planet.id;
        const isSelected = selectedId === node.planet.id;
        const color = tempToColor(node.planet.eqTempK);
        const glowColor = tempToGlow(node.planet.eqTempK);

        if (node.planet.inHabitableZone) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius + 6 * dprRef.current, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(0, 255, 136, ${0.25 + 0.15 * Math.sin(t * 1.5 + node.phase)})`;
          ctx.lineWidth = 2 * dprRef.current;
          ctx.stroke();
        }

        const glowSize = isHovered || isSelected ? 3 : 2;
        const gradient = ctx.createRadialGradient(
          node.x, node.y, node.radius * 0.2,
          node.x, node.y, node.radius * glowSize
        );
        gradient.addColorStop(0, glowColor);
        gradient.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius * glowSize, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);

        const bodyGrad = ctx.createRadialGradient(
          node.x - node.radius * 0.3, node.y - node.radius * 0.3, node.radius * 0.1,
          node.x, node.y, node.radius
        );
        bodyGrad.addColorStop(0, lighten(color, 40));
        bodyGrad.addColorStop(0.7, color);
        bodyGrad.addColorStop(1, darken(color, 30));
        ctx.fillStyle = bodyGrad;
        ctx.fill();

        if (isSelected) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius + 4 * dprRef.current, 0, Math.PI * 2);
          ctx.strokeStyle = "#00d4ff";
          ctx.lineWidth = 2.5 * dprRef.current;
          ctx.stroke();
        }

        if (isHovered) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius + 3 * dprRef.current, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(255,255,255,0.5)";
          ctx.lineWidth = 1.5 * dprRef.current;
          ctx.stroke();
        }

        if (node.planet.hasJWSTData) {
          const bx = node.x + node.radius * 0.7;
          const by = node.y - node.radius * 0.7;
          const bs = 4 * dprRef.current;
          ctx.beginPath();
          ctx.arc(bx, by, bs, 0, Math.PI * 2);
          ctx.fillStyle = "#ffaa00";
          ctx.fill();
          ctx.fillStyle = "#000";
          ctx.font = `bold ${Math.round(5 * dprRef.current)}px Inter, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("J", bx, by + 0.5 * dprRef.current);
        }

        if (!compact || isHovered || isSelected) {
          ctx.fillStyle = isSelected ? "#00d4ff" : isHovered ? "#ffffff" : "rgba(255,255,255,0.6)";
          ctx.font = `${isSelected || isHovered ? "600" : "400"} ${Math.round((compact ? 10 : 11) * dprRef.current)}px Inter, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText(node.planet.name, node.x, node.y + node.radius + 6 * dprRef.current);
        }
      }

      animRef.current = requestAnimationFrame(draw);
    }

    resize();
    animRef.current = requestAnimationFrame(draw);

    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [buildNodes, selectedId]);

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

      {!compact && (
        <>
          <div className="absolute bottom-3 left-4 text-[10px] text-white/30 pointer-events-none select-none">
            Distance (ly) &rarr;
          </div>
          <div className="absolute top-4 left-3 text-[10px] text-white/30 pointer-events-none select-none [writing-mode:vertical-lr] rotate-180">
            Temperature (K) &rarr;
          </div>
        </>
      )}

      {hovered && (
        <div
          className="absolute z-30 pointer-events-none"
          style={{
            left: hovered.x + 16,
            top: hovered.y - 10,
          }}
        >
          <div className="bg-space-800/95 backdrop-blur-md border border-white/10 rounded-xl px-4 py-3 shadow-xl min-w-[180px]">
            <div className="flex items-center gap-2 mb-1.5">
              <div
                className="w-3 h-3 rounded-full"
                style={{ background: tempToColor(hovered.planet.eqTempK) }}
              />
              <span className="font-semibold text-white text-sm">{hovered.planet.name}</span>
              {hovered.planet.hasJWSTData && (
                <Satellite className="w-3 h-3 text-amber" />
              )}
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs text-white/50">
              <span>Temp</span>
              <span className="text-white/80">{hovered.planet.eqTempK ?? "N/A"} K</span>
              <span>Distance</span>
              <span className="text-white/80">{hovered.planet.distanceLy} ly</span>
              <span>Radius</span>
              <span className="text-white/80">{hovered.planet.radiusEarth} R&#8853;</span>
              {hovered.planet.inHabitableZone && (
                <>
                  <span className="col-span-2 text-green mt-1 font-medium">
                    Habitable Zone
                  </span>
                </>
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
  padding: number
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
          const angle = dist === 0 ? Math.random() * Math.PI * 2 : Math.atan2(dy, dx);
          const push = overlap * 0.55;

          a.baseX -= Math.cos(angle) * push;
          a.baseY -= Math.sin(angle) * push;
          b.baseX += Math.cos(angle) * push;
          b.baseY += Math.sin(angle) * push;

          a.baseX = Math.max(padding + a.radius, Math.min(w - padding - a.radius, a.baseX));
          a.baseY = Math.max(padding + a.radius, Math.min(h - padding - a.radius, a.baseY));
          b.baseX = Math.max(padding + b.radius, Math.min(w - padding - b.radius, b.baseX));
          b.baseY = Math.max(padding + b.radius, Math.min(h - padding - b.radius, b.baseY));

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

function lighten(hex: string, pct: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const f = pct / 100;
  return `rgb(${Math.min(255, r + (255 - r) * f)}, ${Math.min(255, g + (255 - g) * f)}, ${Math.min(255, b + (255 - b) * f)})`;
}

function darken(hex: string, pct: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const f = 1 - pct / 100;
  return `rgb(${Math.round(r * f)}, ${Math.round(g * f)}, ${Math.round(b * f)})`;
}
