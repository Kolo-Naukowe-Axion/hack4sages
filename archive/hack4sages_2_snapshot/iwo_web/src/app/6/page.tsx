"use client";

import { Libre_Baskerville, Nunito_Sans, JetBrains_Mono } from "next/font/google";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

const baskerville = Libre_Baskerville({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-baskerville",
});

const nunito = Nunito_Sans({
  subsets: ["latin"],
  weight: ["300", "400", "600", "700", "800"],
  variable: "--font-nunito",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-mono",
});

const NAVY = "#0f2b46";
const ORANGE = "#c85a2a";
const CREAM = "#faf9f7";
const LIGHT_NAVY = "#1a3d5c";
const SOFT_GRAY = "#e8e6e1";
const TEXT_BODY = "#2d2d2d";
const TEXT_MUTED = "#5a5a5a";

const moleculeData = [
  { name: "H₂O", exobiome: 0.218, label: "0.218" },
  { name: "CO₂", exobiome: 0.261, label: "0.261" },
  { name: "CO", exobiome: 0.327, label: "0.327" },
  { name: "CH₄", exobiome: 0.290, label: "0.290" },
  { name: "NH₃", exobiome: 0.378, label: "0.378" },
];

const comparisonData = [
  { name: "Random\nForest", mrmse: 1.2, fill: "#b0b0b0" },
  { name: "CNN\nBaseline", mrmse: 0.85, fill: "#8a8a8a" },
  { name: "ADC\nWinner", mrmse: 0.32, fill: LIGHT_NAVY },
  { name: "ExoBiome", mrmse: 0.295, fill: ORANGE },
];

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <h3
      className="font-bold text-[11px] uppercase tracking-[0.12em] pb-[3px] mb-[6px]"
      style={{
        color: NAVY,
        borderBottom: `2px solid ${ORANGE}`,
        fontFamily: "var(--font-nunito)",
      }}
    >
      {children}
    </h3>
  );
}

function QuantumCircuitDiagram() {
  const qubits = 12;
  const gateW = 22;
  const gateH = 14;
  const rowH = 16;
  const startX = 32;
  const wireLen = 210;
  const svgW = 260;
  const svgH = qubits * rowH + 20;

  const gates: { q: number; x: number; label: string; color: string }[] = [];
  const cnots: { ctrl: number; tgt: number; x: number }[] = [];

  for (let i = 0; i < qubits; i++) {
    gates.push({ q: i, x: 0, label: "Rᵧ", color: "#2563eb" });
  }
  for (let i = 0; i < qubits; i++) {
    gates.push({ q: i, x: 1, label: "Rz", color: "#7c3aed" });
  }
  for (let i = 0; i < qubits - 1; i += 2) {
    cnots.push({ ctrl: i, tgt: i + 1, x: 2 });
  }
  for (let i = 1; i < qubits - 1; i += 2) {
    cnots.push({ ctrl: i, tgt: i + 1, x: 3 });
  }
  for (let i = 0; i < qubits; i++) {
    gates.push({ q: i, x: 4, label: "Rᵧ", color: "#2563eb" });
  }
  for (let i = 0; i < qubits; i++) {
    gates.push({ q: i, x: 5, label: "M", color: "#374151" });
  }

  const colSpacing = 28;

  return (
    <svg
      viewBox={`0 0 ${svgW} ${svgH}`}
      className="w-full"
      style={{ maxHeight: "150px" }}
    >
      {Array.from({ length: qubits }, (_, i) => {
        const y = 10 + i * rowH + rowH / 2;
        return (
          <g key={`wire-${i}`}>
            <line
              x1={4}
              y1={y}
              x2={startX + wireLen}
              y2={y}
              stroke="#c0bdb5"
              strokeWidth={0.6}
            />
            <text
              x={2}
              y={y + 3.5}
              fontSize={7}
              fill={TEXT_MUTED}
              fontFamily="var(--font-mono)"
            >
              q{i}
            </text>
          </g>
        );
      })}

      {gates.map((g, idx) => {
        const x = startX + g.x * colSpacing;
        const y = 10 + g.q * rowH + rowH / 2;
        return (
          <g key={`gate-${idx}`}>
            <rect
              x={x - gateW / 2}
              y={y - gateH / 2}
              width={gateW}
              height={gateH}
              rx={2}
              fill={g.color}
              opacity={0.85}
            />
            <text
              x={x}
              y={y + 3}
              textAnchor="middle"
              fontSize={6.5}
              fill="white"
              fontFamily="var(--font-mono)"
              fontWeight={500}
            >
              {g.label}
            </text>
          </g>
        );
      })}

      {cnots.map((c, idx) => {
        const x = startX + c.x * colSpacing;
        const y1 = 10 + c.ctrl * rowH + rowH / 2;
        const y2 = 10 + c.tgt * rowH + rowH / 2;
        return (
          <g key={`cnot-${idx}`}>
            <line
              x1={x}
              y1={y1}
              x2={x}
              y2={y2}
              stroke={ORANGE}
              strokeWidth={1.2}
            />
            <circle cx={x} cy={y1} r={3} fill={ORANGE} />
            <circle
              cx={x}
              cy={y2}
              r={5}
              fill="none"
              stroke={ORANGE}
              strokeWidth={1.2}
            />
            <line
              x1={x}
              y1={y2 - 5}
              x2={x}
              y2={y2 + 5}
              stroke={ORANGE}
              strokeWidth={1.2}
            />
          </g>
        );
      })}
    </svg>
  );
}

function ArchitectureDiagram() {
  return (
    <svg viewBox="0 0 280 160" className="w-full" style={{ maxHeight: "130px" }}>
      <defs>
        <marker
          id="arrowHead"
          markerWidth="6"
          markerHeight="4"
          refX="5"
          refY="2"
          orient="auto"
        >
          <polygon points="0 0, 6 2, 0 4" fill={NAVY} />
        </marker>
      </defs>

      <rect x="2" y="10" width="62" height="28" rx="4" fill="#e8f0fe" stroke="#2563eb" strokeWidth={0.8} />
      <text x="33" y="22" textAnchor="middle" fontSize="6.5" fill={NAVY} fontFamily="var(--font-nunito)" fontWeight={600}>Spectrum</text>
      <text x="33" y="31" textAnchor="middle" fontSize="5.5" fill={TEXT_MUTED} fontFamily="var(--font-mono)">52 bins</text>

      <rect x="2" y="48" width="62" height="28" rx="4" fill="#fef3e2" stroke={ORANGE} strokeWidth={0.8} />
      <text x="33" y="60" textAnchor="middle" fontSize="6.5" fill={NAVY} fontFamily="var(--font-nunito)" fontWeight={600}>Aux Features</text>
      <text x="33" y="69" textAnchor="middle" fontSize="5.5" fill={TEXT_MUTED} fontFamily="var(--font-mono)">R★, T★, ...</text>

      <line x1="64" y1="24" x2="80" y2="50" stroke={NAVY} strokeWidth={0.8} markerEnd="url(#arrowHead)" />
      <line x1="64" y1="62" x2="80" y2="56" stroke={NAVY} strokeWidth={0.8} markerEnd="url(#arrowHead)" />

      <rect x="82" y="12" width="50" height="24" rx="3" fill="#dbeafe" stroke="#2563eb" strokeWidth={0.7} />
      <text x="107" y="27" textAnchor="middle" fontSize="5.5" fill={NAVY} fontFamily="var(--font-nunito)" fontWeight={600}>SpectralEnc</text>

      <rect x="82" y="48" width="50" height="24" rx="3" fill="#fde8d0" stroke={ORANGE} strokeWidth={0.7} />
      <text x="107" y="63" textAnchor="middle" fontSize="5.5" fill={NAVY} fontFamily="var(--font-nunito)" fontWeight={600}>AuxEncoder</text>

      <line x1="132" y1="24" x2="148" y2="50" stroke={NAVY} strokeWidth={0.8} markerEnd="url(#arrowHead)" />
      <line x1="132" y1="60" x2="148" y2="55" stroke={NAVY} strokeWidth={0.8} markerEnd="url(#arrowHead)" />

      <rect x="150" y="40" width="42" height="24" rx="3" fill="#e0e7ef" stroke={NAVY} strokeWidth={0.8} />
      <text x="171" y="55" textAnchor="middle" fontSize="6" fill={NAVY} fontFamily="var(--font-nunito)" fontWeight={700}>Fusion</text>

      <line x1="192" y1="52" x2="208" y2="52" stroke={NAVY} strokeWidth={0.8} markerEnd="url(#arrowHead)" />

      <rect x="210" y="30" width="60" height="44" rx="5" fill="#f0e6ff" stroke="#7c3aed" strokeWidth={1} />
      <text x="240" y="46" textAnchor="middle" fontSize="6.5" fill="#5b21b6" fontFamily="var(--font-nunito)" fontWeight={700}>Quantum</text>
      <text x="240" y="55" textAnchor="middle" fontSize="6.5" fill="#5b21b6" fontFamily="var(--font-nunito)" fontWeight={700}>Circuit</text>
      <text x="240" y="66" textAnchor="middle" fontSize="5.5" fill={TEXT_MUTED} fontFamily="var(--font-mono)">12 qubits</text>

      <line x1="240" y1="74" x2="240" y2="90" stroke={NAVY} strokeWidth={0.8} markerEnd="url(#arrowHead)" />

      {moleculeData.map((mol, i) => {
        const x = 178 + i * 26;
        return (
          <g key={mol.name}>
            <rect x={x} y="94" width="24" height="18" rx="3" fill="#fff7ed" stroke={ORANGE} strokeWidth={0.7} />
            <text x={x + 12} y="106" textAnchor="middle" fontSize="6" fill={NAVY} fontFamily="var(--font-mono)" fontWeight={700}>
              {mol.name}
            </text>
          </g>
        );
      })}

      <text x="240" y="126" textAnchor="middle" fontSize="5.5" fill={TEXT_MUTED} fontFamily="var(--font-nunito)">
        log₁₀ VMR predictions
      </text>
    </svg>
  );
}

function CustomBarLabel(props: { x?: number; y?: number; width?: number; value?: number }) {
  const { x = 0, y = 0, width = 0, value } = props;
  return (
    <text
      x={x + width / 2}
      y={y - 3}
      fill={TEXT_BODY}
      textAnchor="middle"
      fontSize={8}
      fontFamily="var(--font-mono)"
      fontWeight={700}
    >
      {value?.toFixed(3)}
    </text>
  );
}

function ComparisonBarLabel(props: { x?: number; y?: number; width?: number; value?: number }) {
  const { x = 0, y = 0, width = 0, value } = props;
  return (
    <text
      x={x + width / 2}
      y={y - 3}
      fill={TEXT_BODY}
      textAnchor="middle"
      fontSize={8}
      fontFamily="var(--font-mono)"
      fontWeight={700}
    >
      {value?.toFixed(2)}
    </text>
  );
}

const moleculeColors = ["#2563eb", "#059669", "#8b5cf6", "#d97706", "#dc2626"];

export default function ExoBiomePoster() {
  return (
    <div
      className={`${baskerville.variable} ${nunito.variable} ${mono.variable}`}
      style={{
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        backgroundColor: CREAM,
        display: "flex",
        flexDirection: "column",
        fontFamily: "var(--font-nunito)",
        color: TEXT_BODY,
      }}
    >
      {/* ═══ TOP NAVY STRIP ═══ */}
      <div
        style={{
          height: "4px",
          background: `linear-gradient(90deg, ${NAVY} 0%, ${LIGHT_NAVY} 40%, ${ORANGE} 100%)`,
          flexShrink: 0,
        }}
      />

      {/* ═══ HEADER BANNER ═══ */}
      <header
        style={{
          padding: "10px 28px 8px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: `1px solid ${SOFT_GRAY}`,
          flexShrink: 0,
        }}
      >
        <div style={{ flex: 1 }}>
          <h1
            style={{
              fontFamily: "var(--font-baskerville)",
              fontSize: "26px",
              fontWeight: 700,
              color: NAVY,
              lineHeight: 1.1,
              letterSpacing: "-0.01em",
            }}
          >
            ExoBiome{" "}
            <span style={{ color: ORANGE, fontSize: "22px" }}>—</span>{" "}
            <span style={{ fontWeight: 400, fontSize: "20px", color: LIGHT_NAVY }}>
              Quantum-Enhanced Biosignature Detection
            </span>
          </h1>
          <p
            style={{
              fontFamily: "var(--font-baskerville)",
              fontSize: "13px",
              color: LIGHT_NAVY,
              marginTop: "1px",
              fontStyle: "italic",
            }}
          >
            from Exoplanet Transmission Spectra
          </p>
          <p
            style={{
              fontSize: "10px",
              color: TEXT_MUTED,
              marginTop: "4px",
              fontFamily: "var(--font-nunito)",
              fontWeight: 400,
              letterSpacing: "0.02em",
            }}
          >
            Iwo Odya&ensp;·&ensp;Michał Szczesny&ensp;·&ensp;Patrycja Wyrwas&ensp;·&ensp;Jan Poręba
          </p>
        </div>

        <div style={{ textAlign: "right", flexShrink: 0, marginLeft: "20px" }}>
          <div
            style={{
              fontSize: "11px",
              fontWeight: 700,
              color: NAVY,
              fontFamily: "var(--font-nunito)",
              letterSpacing: "0.06em",
              textTransform: "uppercase",
            }}
          >
            HACK-4-SAGES 2026
          </div>
          <div
            style={{
              fontSize: "9px",
              color: TEXT_MUTED,
              fontFamily: "var(--font-nunito)",
              marginTop: "2px",
            }}
          >
            ETH Zurich · Origins Federation
          </div>
          <div
            style={{
              fontSize: "9px",
              color: TEXT_MUTED,
              fontFamily: "var(--font-nunito)",
            }}
          >
            Life Detection &amp; Biosignatures
          </div>
          <div
            style={{
              display: "flex",
              gap: "8px",
              marginTop: "5px",
              justifyContent: "flex-end",
              alignItems: "center",
            }}
          >
            {["ETH Zurich", "PWR", "IQM"].map((logo) => (
              <div
                key={logo}
                style={{
                  padding: "2px 8px",
                  border: `1px solid ${SOFT_GRAY}`,
                  borderRadius: "3px",
                  fontSize: "7.5px",
                  color: TEXT_MUTED,
                  fontFamily: "var(--font-mono)",
                  fontWeight: 500,
                }}
              >
                {logo}
              </div>
            ))}
          </div>
        </div>
      </header>

      {/* ═══ MAIN 3-COLUMN BODY ═══ */}
      <main
        style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: "1fr 1.15fr 1fr",
          gap: "16px",
          padding: "10px 24px 6px",
          minHeight: 0,
          overflow: "hidden",
        }}
      >
        {/* ─── COLUMN 1: INTRODUCTION + METHODS ─── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "10px", overflow: "hidden" }}>
          <section>
            <SectionHeader>Introduction</SectionHeader>
            <p style={{ fontSize: "10px", lineHeight: 1.55, color: TEXT_BODY, textAlign: "justify" }}>
              Detecting biosignatures in exoplanet atmospheres is a central
              challenge in astrobiology. Transmission spectroscopy during
              planetary transits reveals molecular absorption features, but
              retrieving accurate volume mixing ratios (VMRs) from noisy,
              degenerate spectra remains computationally expensive with
              traditional Bayesian methods.
            </p>
            <p style={{ fontSize: "10px", lineHeight: 1.55, color: TEXT_BODY, marginTop: "5px", textAlign: "justify" }}>
              We present <strong style={{ color: ORANGE }}>ExoBiome</strong>, a
              hybrid quantum-classical neural network that predicts log₁₀ VMR
              for five key biosignature molecules:{" "}
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "9px" }}>
                H₂O, CO₂, CO, CH₄, NH₃
              </span>
              . By embedding a parameterized quantum circuit into the inference
              pipeline, ExoBiome exploits quantum feature spaces to capture
              correlations inaccessible to purely classical models.
            </p>
          </section>

          <section>
            <SectionHeader>Methods</SectionHeader>
            <p style={{ fontSize: "10px", lineHeight: 1.55, color: TEXT_BODY, textAlign: "justify" }}>
              <strong>Data.</strong> We train on the Ariel Data Challenge 2023
              dataset (41,000 simulated transmission spectra, 52 wavelength
              bins, 0.5–7.8 μm) augmented with auxiliary stellar/planetary
              parameters (R★, T★, M_p, R_p, etc.).
            </p>
            <p style={{ fontSize: "10px", lineHeight: 1.55, color: TEXT_BODY, marginTop: "5px", textAlign: "justify" }}>
              <strong>Preprocessing.</strong> Per-bin standardization with robust
              scaling. Auxiliary features undergo log-transform where
              appropriate. Train/validation/test split: 70/15/15.
            </p>
            <p style={{ fontSize: "10px", lineHeight: 1.55, color: TEXT_BODY, marginTop: "5px", textAlign: "justify" }}>
              <strong>Architecture.</strong> A SpectralEncoder (1D-CNN with
              residual connections) processes the 52-bin spectrum. An AuxEncoder
              (MLP) processes auxiliary features. Both are fused and projected
              into the quantum circuit input space.
            </p>
            <p style={{ fontSize: "10px", lineHeight: 1.55, color: TEXT_BODY, marginTop: "5px", textAlign: "justify" }}>
              <strong>Training.</strong> Smooth L1 loss with cosine annealing
              scheduler. Gradient-based optimization of both classical and
              quantum parameters via parameter-shift rule. Total parameters: ~48K classical + 72 quantum.
            </p>
          </section>

          <section>
            <SectionHeader>Key Innovation</SectionHeader>
            <div
              style={{
                background: "linear-gradient(135deg, #f0e6ff 0%, #e8f4fd 100%)",
                borderRadius: "6px",
                padding: "8px 10px",
                border: "1px solid #d4c5f9",
              }}
            >
              <p style={{ fontSize: "10px", lineHeight: 1.5, color: NAVY }}>
                The quantum circuit acts as a <strong>trainable kernel</strong> in
                Hilbert space. A 12-qubit parameterized ansatz with{" "}
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "9px" }}>
                  Rᵧ–Rz
                </span>{" "}
                rotation layers and entangling CNOT gates maps classical features
                into an exponentially large feature space (2¹² = 4096 dimensions),
                enabling the model to capture multi-molecular correlations that
                classical networks miss.
              </p>
            </div>
          </section>
        </div>

        {/* ─── COLUMN 2: ARCHITECTURE + CIRCUIT ─── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "10px", overflow: "hidden" }}>
          <section>
            <SectionHeader>Model Architecture</SectionHeader>
            <div
              style={{
                background: "white",
                borderRadius: "6px",
                padding: "8px 6px 4px",
                border: `1px solid ${SOFT_GRAY}`,
              }}
            >
              <ArchitectureDiagram />
            </div>
          </section>

          <section>
            <SectionHeader>Quantum Circuit (12-Qubit Ansatz)</SectionHeader>
            <div
              style={{
                background: "white",
                borderRadius: "6px",
                padding: "6px",
                border: `1px solid ${SOFT_GRAY}`,
              }}
            >
              <QuantumCircuitDiagram />
              <div
                style={{
                  display: "flex",
                  gap: "12px",
                  justifyContent: "center",
                  marginTop: "4px",
                  flexWrap: "wrap",
                }}
              >
                {[
                  { color: "#2563eb", label: "Rᵧ rotation" },
                  { color: "#7c3aed", label: "Rz rotation" },
                  { color: ORANGE, label: "CNOT entangling" },
                  { color: "#374151", label: "Measurement" },
                ].map((item) => (
                  <div
                    key={item.label}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "4px",
                    }}
                  >
                    <div
                      style={{
                        width: "8px",
                        height: "8px",
                        borderRadius: "2px",
                        backgroundColor: item.color,
                        opacity: 0.85,
                      }}
                    />
                    <span
                      style={{
                        fontSize: "7.5px",
                        color: TEXT_MUTED,
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      {item.label}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section>
            <SectionHeader>Performance Comparison (mRMSE ↓)</SectionHeader>
            <div
              style={{
                background: "white",
                borderRadius: "6px",
                padding: "6px 2px 2px 0",
                border: `1px solid ${SOFT_GRAY}`,
                height: "140px",
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={comparisonData}
                  margin={{ top: 16, right: 12, left: -8, bottom: 4 }}
                  barCategoryGap="28%"
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#e5e5e5"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="name"
                    tick={{
                      fontSize: 8,
                      fill: TEXT_BODY,
                      fontFamily: "var(--font-nunito)",
                    }}
                    tickLine={false}
                    axisLine={{ stroke: "#d0d0d0" }}
                    interval={0}
                  />
                  <YAxis
                    domain={[0, 1.4]}
                    tick={{
                      fontSize: 7,
                      fill: TEXT_MUTED,
                      fontFamily: "var(--font-mono)",
                    }}
                    tickLine={false}
                    axisLine={false}
                    tickCount={5}
                  />
                  <Tooltip
                    contentStyle={{
                      fontSize: "10px",
                      fontFamily: "var(--font-mono)",
                      borderRadius: "4px",
                      border: `1px solid ${SOFT_GRAY}`,
                    }}
                    formatter={(value) => [Number(value).toFixed(3), "mRMSE"]}
                  />
                  <ReferenceLine
                    y={0.295}
                    stroke={ORANGE}
                    strokeDasharray="4 3"
                    strokeWidth={1}
                    label={{
                      value: "ExoBiome: 0.295",
                      position: "right",
                      fontSize: 7,
                      fill: ORANGE,
                      fontFamily: "var(--font-mono)",
                    }}
                  />
                  <Bar
                    dataKey="mrmse"
                    radius={[3, 3, 0, 0]}
                    label={<ComparisonBarLabel />}
                  >
                    {comparisonData.map((entry, index) => (
                      <Cell key={index} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        </div>

        {/* ─── COLUMN 3: RESULTS ─── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "10px", overflow: "hidden" }}>
          <section>
            <SectionHeader>Results — Per-Molecule mRMSE</SectionHeader>
            <div
              style={{
                background: "white",
                borderRadius: "6px",
                padding: "6px 2px 2px 0",
                border: `1px solid ${SOFT_GRAY}`,
                height: "140px",
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={moleculeData}
                  margin={{ top: 16, right: 12, left: -8, bottom: 4 }}
                  barCategoryGap="20%"
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#e5e5e5"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="name"
                    tick={{
                      fontSize: 9,
                      fill: TEXT_BODY,
                      fontFamily: "var(--font-mono)",
                      fontWeight: 700,
                    }}
                    tickLine={false}
                    axisLine={{ stroke: "#d0d0d0" }}
                  />
                  <YAxis
                    domain={[0, 0.5]}
                    tick={{
                      fontSize: 7,
                      fill: TEXT_MUTED,
                      fontFamily: "var(--font-mono)",
                    }}
                    tickLine={false}
                    axisLine={false}
                    tickCount={6}
                  />
                  <Tooltip
                    contentStyle={{
                      fontSize: "10px",
                      fontFamily: "var(--font-mono)",
                      borderRadius: "4px",
                      border: `1px solid ${SOFT_GRAY}`,
                    }}
                    formatter={(value) => [Number(value).toFixed(3), "mRMSE"]}
                  />
                  <Bar dataKey="exobiome" radius={[3, 3, 0, 0]} label={<CustomBarLabel />}>
                    {moleculeData.map((_, index) => (
                      <Cell key={index} fill={moleculeColors[index]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section>
            <SectionHeader>Detailed Results</SectionHeader>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "9px",
                fontFamily: "var(--font-mono)",
              }}
            >
              <thead>
                <tr style={{ borderBottom: `2px solid ${NAVY}` }}>
                  <th
                    style={{
                      textAlign: "left",
                      padding: "3px 6px",
                      fontFamily: "var(--font-nunito)",
                      fontWeight: 700,
                      color: NAVY,
                      fontSize: "9px",
                    }}
                  >
                    Molecule
                  </th>
                  <th
                    style={{
                      textAlign: "center",
                      padding: "3px 6px",
                      fontFamily: "var(--font-nunito)",
                      fontWeight: 700,
                      color: NAVY,
                      fontSize: "9px",
                    }}
                  >
                    mRMSE
                  </th>
                  <th
                    style={{
                      textAlign: "center",
                      padding: "3px 6px",
                      fontFamily: "var(--font-nunito)",
                      fontWeight: 700,
                      color: NAVY,
                      fontSize: "9px",
                    }}
                  >
                    vs ADC Winner
                  </th>
                  <th
                    style={{
                      textAlign: "center",
                      padding: "3px 6px",
                      fontFamily: "var(--font-nunito)",
                      fontWeight: 700,
                      color: NAVY,
                      fontSize: "9px",
                    }}
                  >
                    Rank
                  </th>
                </tr>
              </thead>
              <tbody>
                {[
                  { mol: "H₂O", rmse: "0.218", vs: "−32%", rank: "Best" },
                  { mol: "CO₂", rmse: "0.261", vs: "−18%", rank: "Best" },
                  { mol: "CO", rmse: "0.327", vs: "−2%", rank: "~1st" },
                  { mol: "CH₄", rmse: "0.290", vs: "−9%", rank: "Best" },
                  { mol: "NH₃", rmse: "0.378", vs: "+18%", rank: "2nd" },
                ].map((row, i) => (
                  <tr
                    key={row.mol}
                    style={{
                      borderBottom: `1px solid ${SOFT_GRAY}`,
                      backgroundColor: i % 2 === 0 ? "transparent" : "#f7f6f3",
                    }}
                  >
                    <td
                      style={{
                        padding: "3px 6px",
                        fontWeight: 700,
                        color: moleculeColors[i],
                      }}
                    >
                      {row.mol}
                    </td>
                    <td
                      style={{
                        textAlign: "center",
                        padding: "3px 6px",
                        fontWeight: 700,
                      }}
                    >
                      {row.rmse}
                    </td>
                    <td
                      style={{
                        textAlign: "center",
                        padding: "3px 6px",
                        color: row.vs.startsWith("−") ? "#059669" : "#dc2626",
                        fontWeight: 600,
                      }}
                    >
                      {row.vs}
                    </td>
                    <td
                      style={{
                        textAlign: "center",
                        padding: "3px 6px",
                        fontWeight: 600,
                        color: row.rank === "Best" ? ORANGE : TEXT_MUTED,
                      }}
                    >
                      {row.rank}
                    </td>
                  </tr>
                ))}
                <tr style={{ borderTop: `2px solid ${NAVY}` }}>
                  <td
                    style={{
                      padding: "4px 6px",
                      fontWeight: 800,
                      color: NAVY,
                      fontFamily: "var(--font-nunito)",
                    }}
                  >
                    Mean
                  </td>
                  <td
                    style={{
                      textAlign: "center",
                      padding: "4px 6px",
                      fontWeight: 800,
                      color: ORANGE,
                      fontSize: "10px",
                    }}
                  >
                    0.295
                  </td>
                  <td
                    style={{
                      textAlign: "center",
                      padding: "4px 6px",
                      fontWeight: 700,
                      color: "#059669",
                    }}
                  >
                    −8%
                  </td>
                  <td
                    style={{
                      textAlign: "center",
                      padding: "4px 6px",
                      fontWeight: 700,
                      color: ORANGE,
                    }}
                  >
                    #1
                  </td>
                </tr>
              </tbody>
            </table>
          </section>

          <section>
            <SectionHeader>Key Metrics</SectionHeader>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "6px",
              }}
            >
              {[
                { label: "Overall mRMSE", value: "0.295", highlight: true },
                { label: "Qubits Used", value: "12", highlight: false },
                { label: "Quantum Params", value: "72", highlight: false },
                { label: "Classical Params", value: "~48K", highlight: false },
                { label: "Training Epochs", value: "6", highlight: false },
                { label: "vs ADC Winner", value: "−8%", highlight: true },
              ].map((metric) => (
                <div
                  key={metric.label}
                  style={{
                    padding: "6px 8px",
                    borderRadius: "4px",
                    background: metric.highlight
                      ? `linear-gradient(135deg, ${NAVY} 0%, ${LIGHT_NAVY} 100%)`
                      : "white",
                    border: metric.highlight ? "none" : `1px solid ${SOFT_GRAY}`,
                    textAlign: "center",
                  }}
                >
                  <div
                    style={{
                      fontSize: "14px",
                      fontWeight: 800,
                      fontFamily: "var(--font-mono)",
                      color: metric.highlight ? ORANGE : NAVY,
                      lineHeight: 1.2,
                    }}
                  >
                    {metric.value}
                  </div>
                  <div
                    style={{
                      fontSize: "7px",
                      color: metric.highlight ? "#a0b4c8" : TEXT_MUTED,
                      fontFamily: "var(--font-nunito)",
                      fontWeight: 600,
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                      marginTop: "1px",
                    }}
                  >
                    {metric.label}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>
      </main>

      {/* ═══ BOTTOM STRIP ═══ */}
      <footer
        style={{
          padding: "6px 28px 8px",
          borderTop: `1px solid ${SOFT_GRAY}`,
          display: "grid",
          gridTemplateColumns: "1.2fr 1fr 1fr",
          gap: "20px",
          flexShrink: 0,
          backgroundColor: "#f5f4f0",
        }}
      >
        <div>
          <h4
            style={{
              fontSize: "9px",
              fontWeight: 800,
              color: NAVY,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              marginBottom: "3px",
              fontFamily: "var(--font-nunito)",
            }}
          >
            Conclusions
          </h4>
          <p style={{ fontSize: "9px", lineHeight: 1.5, color: TEXT_BODY }}>
            ExoBiome achieves <strong style={{ color: ORANGE }}>state-of-the-art mRMSE of 0.295</strong>,
            surpassing the ADC 2023 winner by 8%. The quantum circuit provides a compact,
            expressive feature mapping with only 72 trainable parameters, demonstrating
            that hybrid quantum-classical approaches are viable for atmospheric retrieval
            even on near-term quantum hardware.
          </p>
        </div>

        <div>
          <h4
            style={{
              fontSize: "9px",
              fontWeight: 800,
              color: NAVY,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              marginBottom: "3px",
              fontFamily: "var(--font-nunito)",
            }}
          >
            Technology Stack
          </h4>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
            {[
              "Python",
              "PyTorch",
              "Qiskit",
              "sQUlearn",
              "qiskit-on-iqm",
              "IQM Spark (5q)",
              "scikit-learn",
              "Next.js",
            ].map((tech) => (
              <span
                key={tech}
                style={{
                  padding: "1px 6px",
                  borderRadius: "3px",
                  fontSize: "7.5px",
                  fontFamily: "var(--font-mono)",
                  fontWeight: 500,
                  color: NAVY,
                  backgroundColor: "#e8e6e1",
                  border: `0.5px solid #d0cec8`,
                }}
              >
                {tech}
              </span>
            ))}
          </div>
        </div>

        <div>
          <h4
            style={{
              fontSize: "9px",
              fontWeight: 800,
              color: NAVY,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              marginBottom: "3px",
              fontFamily: "var(--font-nunito)",
            }}
          >
            References
          </h4>
          <div style={{ fontSize: "7.5px", lineHeight: 1.55, color: TEXT_MUTED }}>
            <p>[1] Vetrano et al. 2025, arXiv:2509.03617 — QELM for atmospheric retrieval</p>
            <p>[2] Changeat et al. 2022, ApJ — ADC 2023 dataset &amp; baseline</p>
            <p>[3] Schwieterman et al. 2018, Astrobiology — Biosignature review</p>
            <p>[4] Seeburger et al. 2023, ApJ — Methanogenesis to planetary spectra</p>
          </div>
        </div>
      </footer>

      {/* ═══ BOTTOM NAVY STRIP ═══ */}
      <div
        style={{
          height: "3px",
          background: `linear-gradient(90deg, ${ORANGE} 0%, ${LIGHT_NAVY} 60%, ${NAVY} 100%)`,
          flexShrink: 0,
        }}
      />
    </div>
  );
}
