"use client";

import { Source_Sans_3, Source_Code_Pro } from "next/font/google";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
} from "recharts";

const sans = Source_Sans_3({
  subsets: ["latin"],
  weight: ["300", "400", "600", "700"],
  variable: "--font-sans",
});

const mono = Source_Code_Pro({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
});

const STREAMLIT_RED = "#ff4b4b";
const STREAMLIT_RED_LIGHT = "#ff4b4b22";
const STREAMLIT_BLUE = "#0068c9";
const STREAMLIT_GREEN = "#09ab3b";
const STREAMLIT_YELLOW = "#faca2b";
const STREAMLIT_BORDER = "#e6eaf1";
const STREAMLIT_BG_SIDEBAR = "#f0f2f6";
const STREAMLIT_TEXT = "#31333f";
const STREAMLIT_TEXT_LIGHT = "#808495";
const STREAMLIT_SUCCESS_BG = "#dff0d8";
const STREAMLIT_SUCCESS_BORDER = "#09ab3b";
const STREAMLIT_INFO_BG = "#d1ecf1";
const STREAMLIT_INFO_BORDER = "#0068c9";
const STREAMLIT_WARNING_BG = "#fff3cd";
const STREAMLIT_WARNING_BORDER = "#faca2b";

const ALL_MOLECULES = ["H₂O", "CO₂", "CO", "CH₄", "NH₃"];

const MOLECULE_COLORS: Record<string, string> = {
  "H₂O": "#1f77b4",
  "CO₂": "#ff7f0e",
  CO: "#2ca02c",
  "CH₄": "#d62728",
  "NH₃": "#9467bd",
};

const MOLECULE_RESULTS: Record<string, number> = {
  "H₂O": 0.218,
  "CO₂": 0.261,
  CO: 0.327,
  "CH₄": 0.29,
  "NH₃": 0.378,
};

const MODEL_COMPARISON = [
  { model: "ExoBiome", mrmse: 0.295, fill: STREAMLIT_RED },
  { model: "ADC Winner", mrmse: 0.32, fill: "#636efa" },
  { model: "CNN Baseline", mrmse: 0.85, fill: "#ab63fa" },
  { model: "RF Baseline", mrmse: 1.2, fill: "#00cc96" },
];

const PER_MOLECULE_DATA = ALL_MOLECULES.map((mol) => ({
  molecule: mol,
  ExoBiome: MOLECULE_RESULTS[mol],
  "CNN Baseline": mol === "H₂O" ? 0.72 : mol === "CO₂" ? 0.81 : mol === "CO" ? 0.94 : mol === "CH₄" ? 0.88 : 0.91,
  "RF Baseline": mol === "H₂O" ? 1.05 : mol === "CO₂" ? 1.12 : mol === "CO" ? 1.35 : mol === "CH₄" ? 1.18 : 1.28,
}));

function StreamlitSpinner({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-3 py-3 px-4">
      <motion.div
        className="w-5 h-5 rounded-full border-2 border-t-transparent"
        style={{ borderColor: `${STREAMLIT_RED} transparent ${STREAMLIT_RED} ${STREAMLIT_RED}` }}
        animate={{ rotate: 360 }}
        transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
      />
      <span className="text-sm" style={{ color: STREAMLIT_TEXT_LIGHT }}>{text}</span>
    </div>
  );
}

function StreamlitMetric({
  label,
  value,
  delta,
  deltaColor = "green",
  delay = 0,
}: {
  label: string;
  value: string;
  delta: string;
  deltaColor?: "green" | "red" | "neutral";
  delay?: number;
}) {
  const colorMap = {
    green: STREAMLIT_GREEN,
    red: STREAMLIT_RED,
    neutral: STREAMLIT_TEXT_LIGHT,
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className="flex flex-col gap-0.5"
    >
      <span
        className="text-sm font-semibold tracking-wide uppercase"
        style={{ color: STREAMLIT_TEXT_LIGHT, fontSize: "0.8rem", letterSpacing: "0.02em" }}
      >
        {label}
      </span>
      <span
        className="text-3xl font-bold"
        style={{ color: STREAMLIT_TEXT, fontFamily: "var(--font-sans)" }}
      >
        {value}
      </span>
      <span className="text-sm font-medium" style={{ color: colorMap[deltaColor] }}>
        {deltaColor === "green" ? "↓ " : deltaColor === "red" ? "↑ " : ""}
        {delta}
      </span>
    </motion.div>
  );
}

function StreamlitExpander({
  title,
  children,
  defaultOpen = false,
  delay = 0,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  delay?: number;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay }}
      className="rounded-lg overflow-hidden"
      style={{ border: `1px solid ${STREAMLIT_BORDER}` }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-50 transition-colors"
        style={{ fontFamily: "var(--font-sans)" }}
      >
        <span className="font-semibold text-sm" style={{ color: STREAMLIT_TEXT }}>
          {title}
        </span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-gray-400"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </motion.span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1" style={{ borderTop: `1px solid ${STREAMLIT_BORDER}` }}>
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function StreamlitCallout({
  type,
  children,
  delay = 0,
}: {
  type: "info" | "success" | "warning";
  children: React.ReactNode;
  delay?: number;
}) {
  const config = {
    info: {
      bg: STREAMLIT_INFO_BG,
      border: STREAMLIT_INFO_BORDER,
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill={STREAMLIT_INFO_BORDER}>
          <circle cx="12" cy="12" r="10" fill="none" stroke={STREAMLIT_INFO_BORDER} strokeWidth="2" />
          <line x1="12" y1="16" x2="12" y2="12" stroke={STREAMLIT_INFO_BORDER} strokeWidth="2" />
          <circle cx="12" cy="8" r="1" fill={STREAMLIT_INFO_BORDER} />
        </svg>
      ),
    },
    success: {
      bg: STREAMLIT_SUCCESS_BG,
      border: STREAMLIT_SUCCESS_BORDER,
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={STREAMLIT_SUCCESS_BORDER} strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <polyline points="9 12 11 14 15 10" />
        </svg>
      ),
    },
    warning: {
      bg: STREAMLIT_WARNING_BG,
      border: STREAMLIT_WARNING_BORDER,
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={STREAMLIT_WARNING_BORDER} strokeWidth="2">
          <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          <line x1="12" y1="9" x2="12" y2="13" />
          <circle cx="12" cy="16.5" r="0.5" fill={STREAMLIT_WARNING_BORDER} />
        </svg>
      ),
    },
  };

  const c = config[type];

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, delay }}
      className="flex items-start gap-3 rounded-md px-4 py-3"
      style={{ backgroundColor: c.bg, borderLeft: `4px solid ${c.border}` }}
    >
      <span className="mt-0.5 shrink-0">{c.icon}</span>
      <div className="text-sm leading-relaxed" style={{ color: STREAMLIT_TEXT }}>
        {children}
      </div>
    </motion.div>
  );
}

function SidebarSelectbox({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: STREAMLIT_TEXT_LIGHT }}>
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-md px-3 py-2 text-sm bg-white cursor-pointer focus:outline-none focus:ring-2"
        style={{
          border: `1px solid ${STREAMLIT_BORDER}`,
          color: STREAMLIT_TEXT,
          fontFamily: "var(--font-sans)",
        }}
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}

function SidebarSlider({
  label,
  min,
  max,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-center">
        <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: STREAMLIT_TEXT_LIGHT }}>
          {label}
        </label>
        <span
          className="text-xs font-mono px-2 py-0.5 rounded"
          style={{ backgroundColor: STREAMLIT_RED_LIGHT, color: STREAMLIT_RED, fontFamily: "var(--font-mono)" }}
        >
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, ${STREAMLIT_RED} 0%, ${STREAMLIT_RED} ${
            ((value - min) / (max - min)) * 100
          }%, #ddd ${((value - min) / (max - min)) * 100}%, #ddd 100%)`,
        }}
      />
      <div className="flex justify-between text-xs" style={{ color: STREAMLIT_TEXT_LIGHT }}>
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function SidebarMultiselect({
  label,
  options,
  selected,
  onChange,
}: {
  label: string;
  options: string[];
  selected: string[];
  onChange: (v: string[]) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: STREAMLIT_TEXT_LIGHT }}>
        {label}
      </label>
      <div className="flex flex-wrap gap-1.5">
        {options.map((opt) => {
          const isSelected = selected.includes(opt);
          return (
            <button
              key={opt}
              onClick={() => {
                if (isSelected) {
                  onChange(selected.filter((s) => s !== opt));
                } else {
                  onChange([...selected, opt]);
                }
              }}
              className="flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium transition-all"
              style={{
                backgroundColor: isSelected ? STREAMLIT_RED : "white",
                color: isSelected ? "white" : STREAMLIT_TEXT,
                border: `1px solid ${isSelected ? STREAMLIT_RED : STREAMLIT_BORDER}`,
              }}
            >
              {opt}
              {isSelected && (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function SidebarRadio({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: STREAMLIT_TEXT_LIGHT }}>
        {label}
      </label>
      <div className="flex flex-col gap-1.5">
        {options.map((opt) => (
          <button
            key={opt}
            onClick={() => onChange(opt)}
            className="flex items-center gap-2.5 text-sm text-left py-1 px-1 rounded hover:bg-white/50 transition-colors"
            style={{ color: STREAMLIT_TEXT }}
          >
            <span
              className="w-4 h-4 rounded-full border-2 flex items-center justify-center shrink-0"
              style={{ borderColor: value === opt ? STREAMLIT_RED : "#ccc" }}
            >
              {value === opt && (
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: STREAMLIT_RED }}
                />
              )}
            </span>
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}

function ArchitectureDiagram() {
  const blocks = [
    { label: "Spectrum Input", sub: "52 wavelengths", color: "#e8f4f8", border: "#0068c9" },
    { label: "SpectralEncoder", sub: "Conv1D + BatchNorm", color: "#f0e8f8", border: "#9467bd" },
    { label: "Aux Input", sub: "Star params", color: "#e8f4f8", border: "#0068c9" },
    { label: "AuxEncoder", sub: "MLP 2-layer", color: "#f0e8f8", border: "#9467bd" },
    { label: "Fusion Layer", sub: "Concatenation", color: "#fff3e0", border: "#ff7f0e" },
    { label: "Quantum Circuit", sub: "12 qubits · RY+CNOT", color: "#fce4ec", border: STREAMLIT_RED },
    { label: "Classical Head", sub: "Linear → 5 outputs", color: "#e8f5e9", border: STREAMLIT_GREEN },
    { label: "Output", sub: "log₁₀ VMR × 5", color: "#e8f5e9", border: STREAMLIT_GREEN },
  ];

  return (
    <div className="flex flex-col items-center gap-1 py-2">
      <div className="flex gap-6 items-start">
        <div className="flex flex-col items-center gap-1">
          {blocks.slice(0, 2).map((b, i) => (
            <div key={i} className="flex flex-col items-center">
              <div
                className="px-4 py-2.5 rounded-lg text-center text-xs font-medium"
                style={{ backgroundColor: b.color, border: `2px solid ${b.border}`, color: STREAMLIT_TEXT, minWidth: 140 }}
              >
                <div className="font-semibold">{b.label}</div>
                <div style={{ color: STREAMLIT_TEXT_LIGHT, fontSize: "0.7rem" }}>{b.sub}</div>
              </div>
              {i === 0 && (
                <svg width="2" height="16"><line x1="1" y1="0" x2="1" y2="16" stroke={STREAMLIT_BORDER} strokeWidth="2" /></svg>
              )}
            </div>
          ))}
        </div>
        <div className="flex flex-col items-center gap-1">
          {blocks.slice(2, 4).map((b, i) => (
            <div key={i} className="flex flex-col items-center">
              <div
                className="px-4 py-2.5 rounded-lg text-center text-xs font-medium"
                style={{ backgroundColor: b.color, border: `2px solid ${b.border}`, color: STREAMLIT_TEXT, minWidth: 140 }}
              >
                <div className="font-semibold">{b.label}</div>
                <div style={{ color: STREAMLIT_TEXT_LIGHT, fontSize: "0.7rem" }}>{b.sub}</div>
              </div>
              {i === 0 && (
                <svg width="2" height="16"><line x1="1" y1="0" x2="1" y2="16" stroke={STREAMLIT_BORDER} strokeWidth="2" /></svg>
              )}
            </div>
          ))}
        </div>
      </div>
      <svg width="200" height="24" className="my-0">
        <line x1="50" y1="0" x2="100" y2="20" stroke={STREAMLIT_BORDER} strokeWidth="2" />
        <line x1="150" y1="0" x2="100" y2="20" stroke={STREAMLIT_BORDER} strokeWidth="2" />
      </svg>
      {blocks.slice(4).map((b, i) => (
        <div key={i} className="flex flex-col items-center">
          <div
            className="px-5 py-2.5 rounded-lg text-center text-xs font-medium"
            style={{ backgroundColor: b.color, border: `2px solid ${b.border}`, color: STREAMLIT_TEXT, minWidth: 180 }}
          >
            <div className="font-semibold">{b.label}</div>
            <div style={{ color: STREAMLIT_TEXT_LIGHT, fontSize: "0.7rem" }}>{b.sub}</div>
          </div>
          {i < blocks.length - 5 && (
            <svg width="2" height="12"><line x1="1" y1="0" x2="1" y2="12" stroke={STREAMLIT_BORDER} strokeWidth="2" /></svg>
          )}
        </div>
      ))}
    </div>
  );
}

function ResultsTable({ molecules }: { molecules: string[] }) {
  const getColor = (val: number) => {
    if (val < 0.25) return { bg: "#d4edda", text: "#155724" };
    if (val < 0.3) return { bg: "#fff3cd", text: "#856404" };
    if (val < 0.35) return { bg: "#ffeeba", text: "#856404" };
    return { bg: "#f8d7da", text: "#721c24" };
  };

  return (
    <div className="overflow-hidden rounded-lg" style={{ border: `1px solid ${STREAMLIT_BORDER}` }}>
      <table className="w-full text-sm" style={{ fontFamily: "var(--font-sans)" }}>
        <thead>
          <tr style={{ backgroundColor: STREAMLIT_BG_SIDEBAR }}>
            <th className="text-left px-4 py-2.5 font-semibold text-xs uppercase tracking-wider" style={{ color: STREAMLIT_TEXT_LIGHT }}>
              Molecule
            </th>
            <th className="text-center px-4 py-2.5 font-semibold text-xs uppercase tracking-wider" style={{ color: STREAMLIT_TEXT_LIGHT }}>
              mRMSE
            </th>
            <th className="text-center px-4 py-2.5 font-semibold text-xs uppercase tracking-wider" style={{ color: STREAMLIT_TEXT_LIGHT }}>
              vs ADC Winner
            </th>
            <th className="text-center px-4 py-2.5 font-semibold text-xs uppercase tracking-wider" style={{ color: STREAMLIT_TEXT_LIGHT }}>
              Status
            </th>
          </tr>
        </thead>
        <tbody>
          {molecules.map((mol, i) => {
            const val = MOLECULE_RESULTS[mol];
            const diff = val - 0.32;
            const diffPct = ((diff / 0.32) * 100).toFixed(1);
            const clr = getColor(val);
            return (
              <motion.tr
                key={mol}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: i * 0.05 }}
                className="border-t"
                style={{ borderColor: STREAMLIT_BORDER }}
              >
                <td className="px-4 py-2.5 font-medium" style={{ color: STREAMLIT_TEXT }}>
                  <span className="inline-flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-sm"
                      style={{ backgroundColor: MOLECULE_COLORS[mol] }}
                    />
                    {mol}
                  </span>
                </td>
                <td className="text-center px-4 py-2.5">
                  <span
                    className="inline-block px-2.5 py-0.5 rounded text-xs font-mono font-semibold"
                    style={{ backgroundColor: clr.bg, color: clr.text, fontFamily: "var(--font-mono)" }}
                  >
                    {val.toFixed(3)}
                  </span>
                </td>
                <td className="text-center px-4 py-2.5">
                  <span className="text-xs font-medium" style={{ color: diff < 0 ? STREAMLIT_GREEN : STREAMLIT_RED }}>
                    {diff < 0 ? "↓" : "↑"} {Math.abs(Number(diffPct))}%
                  </span>
                </td>
                <td className="text-center px-4 py-2.5">
                  {val < 0.32 ? (
                    <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ backgroundColor: "#d4edda", color: "#155724" }}>
                      Beats Winner
                    </span>
                  ) : (
                    <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ backgroundColor: "#fff3cd", color: "#856404" }}>
                      Competitive
                    </span>
                  )}
                </td>
              </motion.tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CustomBarTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; name: string; color: string }>; label?: string }) {
  if (!active || !payload) return null;
  return (
    <div
      className="px-3 py-2 rounded-md shadow-lg text-xs"
      style={{ backgroundColor: "white", border: `1px solid ${STREAMLIT_BORDER}`, fontFamily: "var(--font-sans)" }}
    >
      <div className="font-semibold mb-1" style={{ color: STREAMLIT_TEXT }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: p.color }} />
          <span style={{ color: STREAMLIT_TEXT_LIGHT }}>{p.name}:</span>
          <span className="font-mono font-semibold" style={{ color: STREAMLIT_TEXT }}>{p.value.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

export default function Page16() {
  const [selectedModel, setSelectedModel] = useState("ExoBiome");
  const [qubits, setQubits] = useState(12);
  const [selectedMolecules, setSelectedMolecules] = useState<string[]>([...ALL_MOLECULES]);
  const [dataset, setDataset] = useState("ADC 2023");
  const [isRunning, setIsRunning] = useState(false);
  const [hasRun, setHasRun] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleRun = () => {
    setIsRunning(true);
    setHasRun(false);
    setTimeout(() => {
      setIsRunning(false);
      setHasRun(true);
    }, 2200);
  };

  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = `
      input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: ${STREAMLIT_RED};
        cursor: pointer;
        border: 2px solid white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
      }
      input[type="range"]::-moz-range-thumb {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: ${STREAMLIT_RED};
        cursor: pointer;
        border: 2px solid white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
      }
      select:focus {
        ring-color: ${STREAMLIT_RED};
        border-color: ${STREAMLIT_RED};
      }
      ::-webkit-scrollbar {
        width: 6px;
      }
      ::-webkit-scrollbar-track {
        background: transparent;
      }
      ::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 3px;
      }
    `;
    document.head.appendChild(style);
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  const filteredPerMolecule = PER_MOLECULE_DATA.filter((d) =>
    selectedMolecules.includes(d.molecule)
  );

  return (
    <div
      className={`${sans.variable} ${mono.variable} min-h-screen flex flex-col`}
      style={{ fontFamily: "var(--font-sans)", backgroundColor: "#ffffff" }}
    >
      {/* Streamlit Top Bar */}
      <header
        className="sticky top-0 z-50 flex items-center justify-between px-4 h-12"
        style={{
          backgroundColor: "white",
          borderBottom: `1px solid ${STREAMLIT_BORDER}`,
          boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
        }}
      >
        <div className="flex items-center gap-3">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 rounded hover:bg-gray-100 transition-colors"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={STREAMLIT_TEXT} strokeWidth="2">
              {sidebarOpen ? (
                <>
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <line x1="9" y1="3" x2="9" y2="21" />
                </>
              ) : (
                <>
                  <line x1="3" y1="6" x2="21" y2="6" />
                  <line x1="3" y1="12" x2="21" y2="12" />
                  <line x1="3" y1="18" x2="21" y2="18" />
                </>
              )}
            </svg>
          </button>
          <span className="text-sm font-semibold" style={{ color: STREAMLIT_TEXT }}>
            ExoBiome
          </span>
        </div>

        <div className="flex items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 transition-colors"
            style={{
              backgroundColor: "white",
              color: STREAMLIT_TEXT,
              border: `1px solid ${STREAMLIT_BORDER}`,
            }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 20h9" />
              <path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z" />
            </svg>
            Deploy
          </motion.button>
          <button className="p-1.5 rounded hover:bg-gray-100 transition-colors">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={STREAMLIT_TEXT_LIGHT} strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
            </svg>
          </button>
        </div>
      </header>

      <div className="flex flex-1">
        {/* Sidebar */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 300, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.25, ease: "easeInOut" }}
              className="shrink-0 overflow-hidden"
              style={{
                backgroundColor: STREAMLIT_BG_SIDEBAR,
                borderRight: `1px solid ${STREAMLIT_BORDER}`,
              }}
            >
              <div className="w-[300px] h-full overflow-y-auto px-5 py-6 flex flex-col gap-6">
                {/* Streamlit logo area */}
                <div className="flex items-center gap-2 mb-1">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" fill={STREAMLIT_RED} opacity="0.8" />
                    <path d="M2 17l10 5 10-5" stroke={STREAMLIT_RED} strokeWidth="2" fill="none" />
                    <path d="M2 12l10 5 10-5" stroke={STREAMLIT_RED} strokeWidth="2" fill="none" />
                  </svg>
                  <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: STREAMLIT_TEXT_LIGHT }}>
                    Configuration
                  </span>
                </div>

                <div style={{ height: 1, backgroundColor: STREAMLIT_BORDER }} />

                <SidebarSelectbox
                  label="Select Model"
                  options={["ExoBiome", "CNN Baseline", "RF Baseline", "ADC Winner"]}
                  value={selectedModel}
                  onChange={setSelectedModel}
                />

                <SidebarSlider
                  label="Number of Qubits"
                  min={2}
                  max={20}
                  value={qubits}
                  onChange={setQubits}
                />

                <SidebarMultiselect
                  label="Target Molecules"
                  options={ALL_MOLECULES}
                  selected={selectedMolecules}
                  onChange={setSelectedMolecules}
                />

                <SidebarRadio
                  label="Dataset"
                  options={["ADC 2023", "ABC Database", "MultiREx", "Custom"]}
                  value={dataset}
                  onChange={setDataset}
                />

                <div style={{ height: 1, backgroundColor: STREAMLIT_BORDER }} />

                <motion.button
                  whileHover={{ scale: 1.02, backgroundColor: "#e04343" }}
                  whileTap={{ scale: 0.97 }}
                  onClick={handleRun}
                  disabled={isRunning}
                  className="w-full py-2.5 rounded-md text-sm font-semibold text-white transition-colors disabled:opacity-60"
                  style={{ backgroundColor: STREAMLIT_RED }}
                >
                  {isRunning ? (
                    <span className="flex items-center justify-center gap-2">
                      <motion.span
                        animate={{ rotate: 360 }}
                        transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
                        className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full"
                      />
                      Running...
                    </span>
                  ) : (
                    "Run Analysis"
                  )}
                </motion.button>

                <div className="mt-auto pt-4" style={{ borderTop: `1px solid ${STREAMLIT_BORDER}` }}>
                  <div className="text-xs" style={{ color: STREAMLIT_TEXT_LIGHT }}>
                    <div className="font-semibold mb-1">HACK-4-SAGES 2026</div>
                    <div>ETH Zurich &middot; COPL</div>
                    <div className="mt-1 opacity-70">Life Detection & Biosignatures</div>
                  </div>
                </div>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-8 py-8">
            {/* st.title */}
            <motion.h1
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="text-3xl font-bold mb-1"
              style={{ color: STREAMLIT_TEXT, fontFamily: "var(--font-sans)" }}
            >
              ExoBiome: Quantum Biosignature Detection
            </motion.h1>

            {/* st.caption */}
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, delay: 0.1 }}
              className="text-sm mb-6"
              style={{ color: STREAMLIT_TEXT_LIGHT }}
            >
              Quantum-classical hybrid neural network for atmospheric retrieval from exoplanet transmission spectra
            </motion.p>

            <div style={{ height: 1, backgroundColor: STREAMLIT_BORDER, marginBottom: 24 }} />

            {/* Running state */}
            {isRunning && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6"
              >
                <StreamlitSpinner text="Running analysis pipeline..." />
                <div className="mt-2 h-1 rounded-full overflow-hidden" style={{ backgroundColor: "#eee" }}>
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: STREAMLIT_RED }}
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 2, ease: "easeInOut" }}
                  />
                </div>
              </motion.div>
            )}

            {hasRun && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="flex flex-col gap-8"
              >
                {/* st.markdown */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.15 }}
                  className="text-sm leading-relaxed"
                  style={{ color: STREAMLIT_TEXT }}
                >
                  <p className="mb-3">
                    ExoBiome is the first quantum machine learning system applied to biosignature detection
                    in exoplanet atmospheres. The model processes transmission spectra and auxiliary stellar
                    parameters through a hybrid quantum-classical architecture to predict molecular
                    volume mixing ratios (log&#8321;&#8320; VMR) for five key atmospheric species.
                  </p>
                  <p>
                    Currently configured with{" "}
                    <span
                      className="px-1.5 py-0.5 rounded text-xs font-semibold"
                      style={{ backgroundColor: STREAMLIT_RED_LIGHT, color: STREAMLIT_RED, fontFamily: "var(--font-mono)" }}
                    >
                      {qubits} qubits
                    </span>{" "}
                    running on the{" "}
                    <span
                      className="px-1.5 py-0.5 rounded text-xs font-semibold"
                      style={{ backgroundColor: STREAMLIT_RED_LIGHT, color: STREAMLIT_RED, fontFamily: "var(--font-mono)" }}
                    >
                      {dataset}
                    </span>{" "}
                    dataset, evaluating{" "}
                    <span
                      className="px-1.5 py-0.5 rounded text-xs font-semibold"
                      style={{ backgroundColor: STREAMLIT_RED_LIGHT, color: STREAMLIT_RED, fontFamily: "var(--font-mono)" }}
                    >
                      {selectedMolecules.length} molecules
                    </span>
                    .
                  </p>
                </motion.div>

                {/* st.columns — metrics */}
                <div>
                  <h3
                    className="text-xs font-semibold uppercase tracking-wider mb-4"
                    style={{ color: STREAMLIT_TEXT_LIGHT }}
                  >
                    Key Metrics
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <StreamlitMetric label="mRMSE" value="0.295" delta="8% vs ADC Winner" deltaColor="green" delay={0.2} />
                    <StreamlitMetric label="Qubits" value={String(qubits)} delta="IQM Spark" deltaColor="neutral" delay={0.3} />
                    <StreamlitMetric label="Parameters" value="120K" delta="10x fewer than CNN" deltaColor="green" delay={0.4} />
                    <StreamlitMetric label="Training" value="~3 min" delta="Per epoch" deltaColor="neutral" delay={0.5} />
                  </div>
                </div>

                {/* st.expander — Architecture */}
                <StreamlitExpander title="Model Architecture" defaultOpen={false} delay={0.3}>
                  <ArchitectureDiagram />
                  <div className="mt-4 text-xs leading-relaxed" style={{ color: STREAMLIT_TEXT_LIGHT }}>
                    <p className="mb-2">
                      The architecture follows a dual-encoder fusion design. The spectral encoder
                      processes the 52-channel transmission spectrum through Conv1D layers with
                      batch normalization and residual connections. The auxiliary encoder handles
                      stellar parameters (T&#8342;, log g, [Fe/H], R&#8342;) through a 2-layer MLP.
                    </p>
                    <p>
                      Features are fused via concatenation and passed through a parameterized quantum
                      circuit with {qubits} qubits using RY rotation and CNOT entangling gates. The quantum
                      layer output feeds into a classical linear head producing 5 molecular abundance predictions.
                    </p>
                  </div>
                </StreamlitExpander>

                {/* st.plotly_chart — Model comparison */}
                <div>
                  <h3
                    className="text-sm font-semibold mb-4"
                    style={{ color: STREAMLIT_TEXT }}
                  >
                    Model Comparison (mRMSE)
                  </h3>
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.35 }}
                    className="rounded-lg p-4"
                    style={{ border: `1px solid ${STREAMLIT_BORDER}` }}
                  >
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={MODEL_COMPARISON} layout="vertical" margin={{ left: 20, right: 30, top: 5, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#eee" horizontal={false} />
                        <XAxis type="number" domain={[0, 1.4]} tick={{ fontSize: 12, fill: STREAMLIT_TEXT_LIGHT }} tickLine={false} axisLine={{ stroke: STREAMLIT_BORDER }} />
                        <YAxis
                          type="category"
                          dataKey="model"
                          width={100}
                          tick={{ fontSize: 12, fill: STREAMLIT_TEXT }}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip content={<CustomBarTooltip />} />
                        <Bar dataKey="mrmse" radius={[0, 4, 4, 0]} barSize={28}>
                          {MODEL_COMPARISON.map((entry, i) => (
                            <Cell key={i} fill={entry.fill} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </motion.div>
                </div>

                {/* st.info */}
                <StreamlitCallout type="info" delay={0.4}>
                  <strong>Quantum Advantage:</strong> ExoBiome achieves a <strong>8% improvement</strong> over
                  the ADC 2023 competition winner while using 10x fewer trainable parameters. The quantum
                  circuit provides an implicit regularization effect that prevents overfitting on limited
                  training data.
                </StreamlitCallout>

                {/* st.dataframe */}
                <div>
                  <h3
                    className="text-sm font-semibold mb-4"
                    style={{ color: STREAMLIT_TEXT }}
                  >
                    Per-Molecule Results
                  </h3>
                  <ResultsTable molecules={selectedMolecules} />
                </div>

                {/* st.plotly_chart — Per-molecule grouped bar chart */}
                {filteredPerMolecule.length > 0 && (
                  <div>
                    <h3
                      className="text-sm font-semibold mb-4"
                      style={{ color: STREAMLIT_TEXT }}
                    >
                      Per-Molecule Comparison
                    </h3>
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.4, delay: 0.45 }}
                      className="rounded-lg p-4"
                      style={{ border: `1px solid ${STREAMLIT_BORDER}` }}
                    >
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={filteredPerMolecule} margin={{ left: 0, right: 20, top: 10, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#eee" vertical={false} />
                          <XAxis
                            dataKey="molecule"
                            tick={{ fontSize: 12, fill: STREAMLIT_TEXT }}
                            tickLine={false}
                            axisLine={{ stroke: STREAMLIT_BORDER }}
                          />
                          <YAxis
                            tick={{ fontSize: 11, fill: STREAMLIT_TEXT_LIGHT }}
                            tickLine={false}
                            axisLine={false}
                            domain={[0, 1.5]}
                          />
                          <Tooltip content={<CustomBarTooltip />} />
                          <Legend
                            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
                            iconType="square"
                            iconSize={10}
                          />
                          <Bar dataKey="ExoBiome" fill={STREAMLIT_RED} radius={[3, 3, 0, 0]} barSize={24} />
                          <Bar dataKey="CNN Baseline" fill="#ab63fa" radius={[3, 3, 0, 0]} barSize={24} />
                          <Bar dataKey="RF Baseline" fill="#00cc96" radius={[3, 3, 0, 0]} barSize={24} />
                        </BarChart>
                      </ResponsiveContainer>
                    </motion.div>
                  </div>
                )}

                {/* st.success */}
                <StreamlitCallout type="success" delay={0.5}>
                  <strong>All targets below threshold.</strong> ExoBiome outperforms classical baselines on
                  every molecule. H&#8322;O achieves the best retrieval accuracy at 0.218 mRMSE, while NH&#8323;
                  remains the most challenging target at 0.378.
                </StreamlitCallout>

                {/* st.expander — Quantum Hardware */}
                <StreamlitExpander title="Quantum Hardware Details" defaultOpen={false} delay={0.5}>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div
                        className="rounded-lg px-4 py-3"
                        style={{ backgroundColor: STREAMLIT_BG_SIDEBAR }}
                      >
                        <div className="text-xs font-semibold uppercase tracking-wider mb-1" style={{ color: STREAMLIT_TEXT_LIGHT }}>
                          Primary QPU
                        </div>
                        <div className="text-sm font-semibold" style={{ color: STREAMLIT_TEXT }}>Odra 5 (IQM Spark)</div>
                        <div className="text-xs mt-1" style={{ color: STREAMLIT_TEXT_LIGHT }}>
                          5 qubits &middot; PWR Wroclaw &middot; qiskit-on-iqm
                        </div>
                      </div>
                      <div
                        className="rounded-lg px-4 py-3"
                        style={{ backgroundColor: STREAMLIT_BG_SIDEBAR }}
                      >
                        <div className="text-xs font-semibold uppercase tracking-wider mb-1" style={{ color: STREAMLIT_TEXT_LIGHT }}>
                          Extended QPU
                        </div>
                        <div className="text-sm font-semibold" style={{ color: STREAMLIT_TEXT }}>VTT Q50</div>
                        <div className="text-xs mt-1" style={{ color: STREAMLIT_TEXT_LIGHT }}>
                          53 qubits &middot; Finland &middot; Remote access
                        </div>
                      </div>
                    </div>
                    <div
                      className="rounded-md px-4 py-3 text-xs font-mono leading-relaxed"
                      style={{
                        backgroundColor: "#1e1e1e",
                        color: "#d4d4d4",
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      <div style={{ color: "#6a9955" }}># Quantum Circuit Configuration</div>
                      <div>
                        <span style={{ color: "#569cd6" }}>n_qubits</span>
                        <span style={{ color: "#d4d4d4" }}> = </span>
                        <span style={{ color: "#b5cea8" }}>{qubits}</span>
                      </div>
                      <div>
                        <span style={{ color: "#569cd6" }}>encoding</span>
                        <span style={{ color: "#d4d4d4" }}> = </span>
                        <span style={{ color: "#ce9178" }}>&quot;angle_encoding&quot;</span>
                      </div>
                      <div>
                        <span style={{ color: "#569cd6" }}>ansatz</span>
                        <span style={{ color: "#d4d4d4" }}> = </span>
                        <span style={{ color: "#ce9178" }}>&quot;RY_CNOT&quot;</span>
                      </div>
                      <div>
                        <span style={{ color: "#569cd6" }}>layers</span>
                        <span style={{ color: "#d4d4d4" }}> = </span>
                        <span style={{ color: "#b5cea8" }}>4</span>
                      </div>
                      <div>
                        <span style={{ color: "#569cd6" }}>measurement</span>
                        <span style={{ color: "#d4d4d4" }}> = </span>
                        <span style={{ color: "#ce9178" }}>&quot;expectation_Z&quot;</span>
                      </div>
                    </div>
                  </div>
                </StreamlitExpander>

                {/* st.expander — Code snippet */}
                <StreamlitExpander title="Usage Example" defaultOpen={false} delay={0.55}>
                  <div
                    className="rounded-md px-4 py-3 text-xs leading-relaxed overflow-x-auto"
                    style={{
                      backgroundColor: "#1e1e1e",
                      color: "#d4d4d4",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    <div style={{ color: "#c586c0" }}>import</div>
                    <div>
                      <span style={{ color: "#c586c0" }}>from </span>
                      <span style={{ color: "#4ec9b0" }}>exobiome</span>
                      <span style={{ color: "#c586c0" }}> import </span>
                      <span style={{ color: "#9cdcfe" }}>ExoBiomeModel</span>
                    </div>
                    <div className="mt-2">
                      <span style={{ color: "#6a9955" }}># Load pretrained model</span>
                    </div>
                    <div>
                      <span style={{ color: "#9cdcfe" }}>model</span>
                      <span> = </span>
                      <span style={{ color: "#4ec9b0" }}>ExoBiomeModel</span>
                      <span style={{ color: "#d4d4d4" }}>.load_pretrained(</span>
                      <span style={{ color: "#ce9178" }}>&quot;v4_epoch6&quot;</span>
                      <span style={{ color: "#d4d4d4" }}>)</span>
                    </div>
                    <div className="mt-2">
                      <span style={{ color: "#6a9955" }}># Predict molecular abundances</span>
                    </div>
                    <div>
                      <span style={{ color: "#9cdcfe" }}>predictions</span>
                      <span> = </span>
                      <span style={{ color: "#9cdcfe" }}>model</span>
                      <span>.</span>
                      <span style={{ color: "#dcdcaa" }}>predict</span>
                      <span>(</span>
                      <span style={{ color: "#9cdcfe" }}>spectrum</span>
                      <span>, </span>
                      <span style={{ color: "#9cdcfe" }}>aux_features</span>
                      <span>)</span>
                    </div>
                    <div className="mt-2">
                      <span style={{ color: "#6a9955" }}># Output: {`{`}&apos;H2O&apos;: -3.21, &apos;CO2&apos;: -4.56, ...{`}`}</span>
                    </div>
                  </div>
                </StreamlitExpander>

                {/* st.warning */}
                <StreamlitCallout type="warning" delay={0.55}>
                  <strong>Note:</strong> Results shown use simulated quantum circuits. Execution on real
                  quantum hardware (Odra 5 / VTT Q50) introduces additional noise which is compensated
                  by error mitigation techniques. Real-hardware mRMSE may vary by +/- 0.02.
                </StreamlitCallout>

                {/* Pipeline visualization */}
                <div>
                  <h3
                    className="text-sm font-semibold mb-4"
                    style={{ color: STREAMLIT_TEXT }}
                  >
                    Detection Pipeline
                  </h3>
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.6 }}
                    className="rounded-lg p-5 flex items-center justify-between gap-2 overflow-x-auto"
                    style={{ border: `1px solid ${STREAMLIT_BORDER}`, backgroundColor: STREAMLIT_BG_SIDEBAR }}
                  >
                    {[
                      { icon: "🔭", label: "Spectrum", sub: "52ch" },
                      { icon: "⭐", label: "Stellar Params", sub: "4 features" },
                      { icon: "🧠", label: "Neural Encoder", sub: "Conv1D+MLP" },
                      { icon: "⚛️", label: "Quantum Layer", sub: `${qubits}q circuit` },
                      { icon: "🧬", label: "Biosignatures", sub: `${selectedMolecules.length} molecules` },
                    ].map((step, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <motion.div
                          initial={{ scale: 0.8, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ duration: 0.3, delay: 0.6 + i * 0.08 }}
                          className="flex flex-col items-center text-center min-w-[80px]"
                        >
                          <span className="text-2xl mb-1">{step.icon}</span>
                          <span className="text-xs font-semibold" style={{ color: STREAMLIT_TEXT }}>
                            {step.label}
                          </span>
                          <span className="text-[10px]" style={{ color: STREAMLIT_TEXT_LIGHT }}>
                            {step.sub}
                          </span>
                        </motion.div>
                        {i < 4 && (
                          <svg width="24" height="16" className="shrink-0" style={{ color: STREAMLIT_TEXT_LIGHT }}>
                            <line x1="0" y1="8" x2="18" y2="8" stroke="currentColor" strokeWidth="1.5" />
                            <polyline points="14,4 20,8 14,12" fill="none" stroke="currentColor" strokeWidth="1.5" />
                          </svg>
                        )}
                      </div>
                    ))}
                  </motion.div>
                </div>

                {/* st.expander — Dataset Details */}
                <StreamlitExpander title="Dataset Information" defaultOpen={false} delay={0.6}>
                  <div className="space-y-3 text-xs" style={{ color: STREAMLIT_TEXT }}>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded-md p-3" style={{ backgroundColor: STREAMLIT_BG_SIDEBAR }}>
                        <div className="font-semibold">ADC 2023</div>
                        <div style={{ color: STREAMLIT_TEXT_LIGHT }}>41,000 spectra &middot; Ground truth</div>
                      </div>
                      <div className="rounded-md p-3" style={{ backgroundColor: STREAMLIT_BG_SIDEBAR }}>
                        <div className="font-semibold">ABC Database</div>
                        <div style={{ color: STREAMLIT_TEXT_LIGHT }}>106,000 spectra &middot; Zenodo</div>
                      </div>
                      <div className="rounded-md p-3" style={{ backgroundColor: STREAMLIT_BG_SIDEBAR }}>
                        <div className="font-semibold">MultiREx</div>
                        <div style={{ color: STREAMLIT_TEXT_LIGHT }}>TauREx 3 based &middot; Custom gen</div>
                      </div>
                      <div className="rounded-md p-3" style={{ backgroundColor: STREAMLIT_BG_SIDEBAR }}>
                        <div className="font-semibold">JWST Validation</div>
                        <div style={{ color: STREAMLIT_TEXT_LIGHT }}>K2-18b, WASP-39b &middot; MAST</div>
                      </div>
                    </div>
                    <p className="leading-relaxed" style={{ color: STREAMLIT_TEXT_LIGHT }}>
                      Primary training uses the ADC 2023 competition dataset with 41,000 synthetic
                      transmission spectra generated via TauREx 3. The ABC Database (106K spectra) serves
                      as additional pre-training data with consistent log&#8321;&#8320; VMR format.
                    </p>
                  </div>
                </StreamlitExpander>

                {/* Tech stack badges */}
                <div>
                  <h3
                    className="text-xs font-semibold uppercase tracking-wider mb-3"
                    style={{ color: STREAMLIT_TEXT_LIGHT }}
                  >
                    Tech Stack
                  </h3>
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.4, delay: 0.65 }}
                    className="flex flex-wrap gap-2"
                  >
                    {[
                      "Python",
                      "PyTorch",
                      "Qiskit",
                      "qiskit-on-iqm",
                      "sQUlearn",
                      "scikit-learn",
                      "TauREx 3",
                      "SpectRes",
                      "Jupyter",
                    ].map((tech, i) => (
                      <motion.span
                        key={tech}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.2, delay: 0.65 + i * 0.03 }}
                        className="px-3 py-1 rounded-full text-xs font-medium"
                        style={{
                          backgroundColor: STREAMLIT_BG_SIDEBAR,
                          color: STREAMLIT_TEXT,
                          border: `1px solid ${STREAMLIT_BORDER}`,
                        }}
                      >
                        {tech}
                      </motion.span>
                    ))}
                  </motion.div>
                </div>

                {/* Footer separator */}
                <div style={{ height: 1, backgroundColor: STREAMLIT_BORDER }} />

                {/* Footer */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.4, delay: 0.7 }}
                  className="flex items-center justify-between text-xs pb-8"
                  style={{ color: STREAMLIT_TEXT_LIGHT }}
                >
                  <div className="flex items-center gap-2">
                    <span>Built with</span>
                    <span className="font-semibold" style={{ color: STREAMLIT_RED }}>Streamlit</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span>HACK-4-SAGES 2026</span>
                    <span>&middot;</span>
                    <span>ETH Zurich</span>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
