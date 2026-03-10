"use client";

import { motion } from "framer-motion";

interface Props {
  onComplete?: () => void;
}

const QUBITS = 5;

const RY_DELAYS = [0.3, 0.45, 0.6, 0.75, 0.9];
const CNOT_PAIRS = [
  { control: 0, target: 1, delay: 1.4 },
  { control: 1, target: 2, delay: 1.65 },
  { control: 2, target: 3, delay: 1.9 },
  { control: 3, target: 4, delay: 2.15 },
];
const MEAS_DELAYS = [2.8, 2.9, 3.0, 3.1, 3.2];

const spring = { type: "spring" as const, stiffness: 400, damping: 20 };

export function QuantumCircuit({ onComplete }: Props) {
  const qSpacing = 40;
  const topPad = 30;
  const leftPad = 50;
  const gateW = 36;
  const gateH = 28;
  const ryX = 120;
  const cnotX = 210;
  const measX = 300;
  const svgW = 380;
  const svgH = topPad + (QUBITS - 1) * qSpacing + 30;

  function qY(q: number) {
    return topPad + q * qSpacing;
  }

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="w-full max-w-[500px] mx-auto"
        style={{ overflow: "visible" }}
      >
        {/* Qubit wires */}
        {Array.from({ length: QUBITS }).map((_, i) => (
          <motion.line
            key={`wire-${i}`}
            x1={leftPad}
            y1={qY(i)}
            x2={svgW - 20}
            y2={qY(i)}
            stroke="#2a2d37"
            strokeWidth={1}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.5, delay: i * 0.05 }}
          />
        ))}

        {/* Qubit labels */}
        {Array.from({ length: QUBITS }).map((_, i) => (
          <motion.text
            key={`ql-${i}`}
            x={leftPad - 8}
            y={qY(i) + 4}
            textAnchor="end"
            className="fill-muted text-[10px] font-mono"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 + i * 0.05 }}
          >
            |0⟩
          </motion.text>
        ))}

        {/* Qubit index */}
        {Array.from({ length: QUBITS }).map((_, i) => (
          <motion.text
            key={`qi-${i}`}
            x={leftPad - 30}
            y={qY(i) + 4}
            textAnchor="end"
            className="fill-accent-blue text-[9px] font-mono font-medium"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 + i * 0.05 }}
          >
            q{i}
          </motion.text>
        ))}

        {/* RY rotation gates */}
        {RY_DELAYS.map((delay, i) => (
          <motion.g key={`ry-${i}`}>
            {/* Pulse glow */}
            <motion.rect
              x={ryX - gateW / 2 - 3}
              y={qY(i) - gateH / 2 - 3}
              width={gateW + 6}
              height={gateH + 6}
              rx={6}
              fill="#3b82f6"
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 0.4, 0] }}
              transition={{ delay, duration: 0.5 }}
            />
            <motion.rect
              x={ryX - gateW / 2}
              y={qY(i) - gateH / 2}
              width={gateW}
              height={gateH}
              rx={4}
              fill="#1a1d27"
              stroke="#3b82f6"
              strokeWidth={1.5}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay, ...spring }}
            />
            <motion.text
              x={ryX}
              y={qY(i) + 4}
              textAnchor="middle"
              className="fill-accent-blue text-[10px] font-mono font-semibold"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: delay + 0.1 }}
            >
              RY
            </motion.text>
          </motion.g>
        ))}

        {/* CNOT gates */}
        {CNOT_PAIRS.map((g, i) => (
          <motion.g key={`cnot-${i}`}>
            {/* Control dot */}
            <motion.circle
              cx={cnotX}
              cy={qY(g.control)}
              r={4}
              fill="#3b82f6"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: g.delay, ...spring }}
            />
            {/* Connecting line */}
            <motion.line
              x1={cnotX}
              y1={qY(g.control)}
              x2={cnotX}
              y2={qY(g.target)}
              stroke="#3b82f6"
              strokeWidth={1.5}
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ delay: g.delay + 0.05, duration: 0.3 }}
            />
            {/* Target circle (XOR) */}
            <motion.circle
              cx={cnotX}
              cy={qY(g.target)}
              r={8}
              fill="none"
              stroke="#3b82f6"
              strokeWidth={1.5}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: g.delay + 0.1, ...spring }}
            />
            <motion.line
              x1={cnotX}
              y1={qY(g.target) - 8}
              x2={cnotX}
              y2={qY(g.target) + 8}
              stroke="#3b82f6"
              strokeWidth={1}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: g.delay + 0.15 }}
            />
            <motion.line
              x1={cnotX - 8}
              y1={qY(g.target)}
              x2={cnotX + 8}
              y2={qY(g.target)}
              stroke="#3b82f6"
              strokeWidth={1}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: g.delay + 0.15 }}
            />
          </motion.g>
        ))}

        {/* Measurement gates */}
        {MEAS_DELAYS.map((delay, i) => (
          <motion.g key={`meas-${i}`}>
            {/* Pulse glow */}
            <motion.rect
              x={measX - gateW / 2 - 3}
              y={qY(i) - gateH / 2 - 3}
              width={gateW + 6}
              height={gateH + 6}
              rx={6}
              fill="#22c55e"
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 0.4, 0] }}
              transition={{ delay, duration: 0.5 }}
            />
            <motion.rect
              x={measX - gateW / 2}
              y={qY(i) - gateH / 2}
              width={gateW}
              height={gateH}
              rx={4}
              fill="#1a1d27"
              stroke="#22c55e"
              strokeWidth={1.5}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay, ...spring }}
            />
            {/* Meter arc */}
            <motion.path
              d={`M ${measX - 6} ${qY(i) + 4} A 6 6 0 0 1 ${measX + 6} ${qY(i) + 4}`}
              fill="none"
              stroke="#22c55e"
              strokeWidth={1}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: delay + 0.1 }}
            />
            {/* Meter needle */}
            <motion.line
              x1={measX}
              y1={qY(i) + 4}
              x2={measX + 4}
              y2={qY(i) - 5}
              stroke="#22c55e"
              strokeWidth={1}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: delay + 0.1 }}
            />
          </motion.g>
        ))}

        {/* Stage labels */}
        <motion.text
          x={ryX}
          y={svgH - 2}
          textAnchor="middle"
          className="fill-muted text-[8px] font-mono"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          ENCODE
        </motion.text>
        <motion.text
          x={cnotX}
          y={svgH - 2}
          textAnchor="middle"
          className="fill-muted text-[8px] font-mono"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
        >
          ENTANGLE
        </motion.text>
        <motion.text
          x={measX}
          y={svgH - 2}
          textAnchor="middle"
          className="fill-muted text-[8px] font-mono"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2.9 }}
        >
          MEASURE
        </motion.text>
      </svg>

      {/* Hardware badge */}
      <motion.div
        className="text-center mt-3"
        initial={{ opacity: 0, y: 5 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 3.4 }}
        onAnimationComplete={onComplete}
      >
        <span className="text-[10px] font-mono text-muted bg-surface border border-border rounded-md px-2 py-0.5">
          IQM Spark &middot; 5 qubits
        </span>
      </motion.div>
    </div>
  );
}
