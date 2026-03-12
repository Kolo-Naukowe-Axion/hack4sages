"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Outfit, IBM_Plex_Mono } from "next/font/google";
import { motion, useInView, AnimatePresence } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  LabelList,
} from "recharts";

const outfit = Outfit({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  variable: "--font-outfit",
});

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["300", "400", "500"],
  variable: "--font-ibm-plex-mono",
});

const TOTAL_SLIDES = 9;
const TEAL = "#0d9488";
const DARK = "#1a1a1a";
const MUTED = "#6b7280";
const LIGHT_GRAY = "#f3f4f6";

const EASE = [0.25, 0.46, 0.45, 0.94] as [number, number, number, number];

const fadeUp = {
  hidden: { opacity: 0, y: 40 },
  visible: (i: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.8,
      delay: i * 0.15,
      ease: EASE,
    },
  }),
};

const fadeIn = {
  hidden: { opacity: 0 },
  visible: (i: number = 0) => ({
    opacity: 1,
    transition: {
      duration: 0.8,
      delay: i * 0.15,
      ease: EASE,
    },
  }),
};

function SlideWrapper({
  children,
  id,
}: {
  children: React.ReactNode;
  id: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { amount: 0.5 });

  return (
    <section
      ref={ref}
      id={`slide-${id}`}
      className="h-screen w-full snap-start snap-always flex items-center justify-center relative overflow-hidden"
      style={{ background: "#ffffff" }}
    >
      <AnimatePresence>
        {isInView && (
          <motion.div
            initial="hidden"
            animate="visible"
            className="w-full h-full flex items-center justify-center px-8 md:px-16 lg:px-24"
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}

/* ─── Slide 1: Title ─── */
function SlideTitle() {
  return (
    <SlideWrapper id={1}>
      <div className="text-center">
        <motion.h1
          variants={fadeUp}
          custom={0}
          style={{
            fontFamily: "var(--font-outfit)",
            fontWeight: 300,
            fontSize: "clamp(4rem, 12vw, 10rem)",
            color: DARK,
            letterSpacing: "-0.03em",
            lineHeight: 1,
          }}
        >
          Exo
          <span style={{ color: TEAL }}>Biome</span>
        </motion.h1>
        <motion.div
          variants={fadeUp}
          custom={1}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.75rem, 1.2vw, 1rem)",
            color: MUTED,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            marginTop: "2rem",
          }}
        >
          Quantum Biosignature Detection
        </motion.div>
        <motion.div
          variants={fadeUp}
          custom={2}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.65rem, 1vw, 0.85rem)",
            color: "#9ca3af",
            letterSpacing: "0.15em",
            textTransform: "uppercase",
            marginTop: "0.75rem",
          }}
        >
          HACK-4-SAGES 2026 &middot; ETH Zurich
        </motion.div>
      </div>
    </SlideWrapper>
  );
}

/* ─── Slide 2: Provocative Question ─── */
function SlideQuestion() {
  return (
    <SlideWrapper id={2}>
      <div className="text-center max-w-4xl">
        <motion.h2
          variants={fadeUp}
          custom={0}
          style={{
            fontFamily: "var(--font-outfit)",
            fontWeight: 300,
            fontSize: "clamp(2rem, 5vw, 4.5rem)",
            color: DARK,
            letterSpacing: "-0.02em",
            lineHeight: 1.15,
          }}
        >
          Can we detect life
          <br />
          <span style={{ color: TEAL }}>from starlight?</span>
        </motion.h2>
        <motion.p
          variants={fadeUp}
          custom={1}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.8rem, 1.1vw, 1rem)",
            color: MUTED,
            marginTop: "2.5rem",
            lineHeight: 1.8,
            maxWidth: "36rem",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        >
          When starlight filters through an exoplanet&apos;s atmosphere,
          molecules absorb specific wavelengths &mdash; leaving fingerprints of
          their presence in the transmission spectrum.
        </motion.p>
      </div>
    </SlideWrapper>
  );
}

/* ─── Slide 3: Five Molecules ─── */
const molecules = [
  { formula: "H\u2082O", name: "Water", color: "#0ea5e9" },
  { formula: "CO\u2082", name: "Carbon Dioxide", color: "#f59e0b" },
  { formula: "CO", name: "Carbon Monoxide", color: "#ef4444" },
  { formula: "CH\u2084", name: "Methane", color: "#22c55e" },
  { formula: "NH\u2083", name: "Ammonia", color: "#a855f7" },
];

function SlideMolecules() {
  return (
    <SlideWrapper id={3}>
      <div className="text-center w-full max-w-5xl">
        <motion.p
          variants={fadeUp}
          custom={0}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.7rem, 1vw, 0.85rem)",
            color: MUTED,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            marginBottom: "3rem",
          }}
        >
          Target Biosignatures
        </motion.p>
        <div className="flex flex-wrap items-center justify-center gap-6 md:gap-10 lg:gap-14">
          {molecules.map((mol, i) => (
            <motion.div
              key={mol.formula}
              variants={fadeUp}
              custom={i + 1}
              className="flex flex-col items-center"
            >
              <div
                style={{
                  width: "clamp(80px, 10vw, 120px)",
                  height: "clamp(80px, 10vw, 120px)",
                  borderRadius: "50%",
                  border: `2px solid ${mol.color}20`,
                  background: `${mol.color}08`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginBottom: "1rem",
                }}
              >
                <span
                  style={{
                    fontFamily: "var(--font-outfit)",
                    fontWeight: 400,
                    fontSize: "clamp(1.2rem, 2.5vw, 2rem)",
                    color: mol.color,
                  }}
                >
                  {mol.formula}
                </span>
              </div>
              <span
                style={{
                  fontFamily: "var(--font-ibm-plex-mono)",
                  fontWeight: 300,
                  fontSize: "0.75rem",
                  color: MUTED,
                  letterSpacing: "0.05em",
                }}
              >
                {mol.name}
              </span>
            </motion.div>
          ))}
        </div>
        <motion.p
          variants={fadeUp}
          custom={7}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.75rem, 1vw, 0.9rem)",
            color: MUTED,
            marginTop: "3rem",
            lineHeight: 1.7,
          }}
        >
          Predict log&#x2081;&#x2080; VMR for each molecule from a 52-bin
          transmission spectrum
        </motion.p>
      </div>
    </SlideWrapper>
  );
}

/* ─── Slide 4: Architecture ─── */
function SlideArchitecture() {
  const pipelineSteps = [
    { label: "Spectrum", sub: "52 bins", icon: "〰" },
    { label: "SpectralEncoder", sub: "Conv1D + Attention", icon: "⊞" },
    { label: "Fusion", sub: "Cross-attention", icon: "⊕" },
    { label: "Quantum Circuit", sub: "12 qubits", icon: "◈" },
    { label: "VMR Output", sub: "5 molecules", icon: "◉" },
  ];

  const auxSteps = [
    { label: "Aux Features", sub: "Stellar params", icon: "★" },
    { label: "AuxEncoder", sub: "MLP + LayerNorm", icon: "⊞" },
  ];

  return (
    <SlideWrapper id={4}>
      <div className="w-full max-w-5xl">
        <motion.p
          variants={fadeUp}
          custom={0}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.7rem, 1vw, 0.85rem)",
            color: MUTED,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            textAlign: "center",
            marginBottom: "3rem",
          }}
        >
          Architecture
        </motion.p>

        {/* Main pipeline */}
        <div className="flex items-center justify-center flex-wrap gap-2 md:gap-0">
          {pipelineSteps.map((step, i) => (
            <motion.div
              key={step.label}
              variants={fadeUp}
              custom={i + 1}
              className="flex items-center"
            >
              <div
                className="flex flex-col items-center px-3 md:px-5 py-4"
                style={{ minWidth: "100px" }}
              >
                <span
                  style={{
                    fontSize: "1.5rem",
                    marginBottom: "0.75rem",
                    color: i === 3 ? TEAL : DARK,
                    filter: i === 3 ? "none" : "none",
                  }}
                >
                  {step.icon}
                </span>
                <span
                  style={{
                    fontFamily: "var(--font-outfit)",
                    fontWeight: 400,
                    fontSize: "clamp(0.8rem, 1.1vw, 0.95rem)",
                    color: i === 3 ? TEAL : DARK,
                    textAlign: "center",
                    marginBottom: "0.25rem",
                  }}
                >
                  {step.label}
                </span>
                <span
                  style={{
                    fontFamily: "var(--font-ibm-plex-mono)",
                    fontWeight: 300,
                    fontSize: "0.7rem",
                    color: MUTED,
                    textAlign: "center",
                  }}
                >
                  {step.sub}
                </span>
              </div>
              {i < pipelineSteps.length - 1 && (
                <div
                  className="hidden md:block"
                  style={{
                    width: "40px",
                    height: "1px",
                    background:
                      i === 2
                        ? `linear-gradient(90deg, ${MUTED}40, ${TEAL})`
                        : `${MUTED}30`,
                    position: "relative",
                  }}
                >
                  <div
                    style={{
                      position: "absolute",
                      right: "-3px",
                      top: "-3px",
                      width: "6px",
                      height: "6px",
                      borderRight: `1.5px solid ${i === 2 ? TEAL : MUTED + "40"}`,
                      borderTop: `1.5px solid ${i === 2 ? TEAL : MUTED + "40"}`,
                      transform: "rotate(45deg)",
                    }}
                  />
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Aux branch */}
        <motion.div
          variants={fadeUp}
          custom={7}
          className="flex items-center justify-center mt-6 gap-2 md:gap-0"
        >
          <div
            className="hidden md:block"
            style={{
              width: "1px",
              height: "30px",
              background: `${MUTED}20`,
              marginRight: "auto",
              marginLeft: "auto",
            }}
          />
        </motion.div>

        <motion.div
          variants={fadeUp}
          custom={7}
          className="flex items-center justify-center gap-2 md:gap-0"
        >
          {auxSteps.map((step, i) => (
            <div key={step.label} className="flex items-center">
              <div className="flex flex-col items-center px-3 md:px-5 py-3">
                <span
                  style={{
                    fontSize: "1.2rem",
                    marginBottom: "0.5rem",
                    color: MUTED,
                  }}
                >
                  {step.icon}
                </span>
                <span
                  style={{
                    fontFamily: "var(--font-outfit)",
                    fontWeight: 400,
                    fontSize: "clamp(0.75rem, 1vw, 0.85rem)",
                    color: MUTED,
                    textAlign: "center",
                    marginBottom: "0.25rem",
                  }}
                >
                  {step.label}
                </span>
                <span
                  style={{
                    fontFamily: "var(--font-ibm-plex-mono)",
                    fontWeight: 300,
                    fontSize: "0.65rem",
                    color: "#9ca3af",
                    textAlign: "center",
                  }}
                >
                  {step.sub}
                </span>
              </div>
              {i < auxSteps.length - 1 && (
                <div
                  className="hidden md:block"
                  style={{
                    width: "30px",
                    height: "1px",
                    background: `${MUTED}30`,
                  }}
                />
              )}
            </div>
          ))}
          <div
            className="hidden md:flex items-center"
            style={{ marginLeft: "0.5rem" }}
          >
            <div
              style={{
                width: "40px",
                height: "1px",
                background: `${MUTED}30`,
              }}
            />
            <span
              style={{
                fontFamily: "var(--font-ibm-plex-mono)",
                fontWeight: 300,
                fontSize: "0.65rem",
                color: "#9ca3af",
                marginLeft: "0.5rem",
              }}
            >
              &uarr; merges at Fusion
            </span>
          </div>
        </motion.div>
      </div>
    </SlideWrapper>
  );
}

/* ─── Slide 5: The Number ─── */
function SlideNumber() {
  return (
    <SlideWrapper id={5}>
      <div className="text-center">
        <motion.p
          variants={fadeUp}
          custom={0}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.7rem, 1vw, 0.85rem)",
            color: MUTED,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            marginBottom: "1.5rem",
          }}
        >
          Mean Relative RMSE
        </motion.p>
        <motion.div
          variants={fadeUp}
          custom={1}
          style={{
            fontFamily: "var(--font-outfit)",
            fontWeight: 300,
            fontSize: "clamp(6rem, 20vw, 16rem)",
            color: TEAL,
            letterSpacing: "-0.04em",
            lineHeight: 1,
          }}
        >
          0.295
        </motion.div>
        <motion.p
          variants={fadeUp}
          custom={2}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.75rem, 1.1vw, 0.95rem)",
            color: MUTED,
            marginTop: "2rem",
            lineHeight: 1.7,
          }}
        >
          Outperforming the ADC 2023 winning solution
        </motion.p>
      </div>
    </SlideWrapper>
  );
}

/* ─── Slide 6: Comparison Chart ─── */
const comparisonData = [
  { name: "Random Forest", mrmse: 1.2, fill: "#d1d5db" },
  { name: "CNN Baseline", mrmse: 0.85, fill: "#9ca3af" },
  { name: "ADC Winner", mrmse: 0.32, fill: "#6b7280" },
  { name: "ExoBiome", mrmse: 0.295, fill: TEAL },
];

function SlideComparison() {
  const ref = useRef(null);
  const isInView = useInView(ref, { amount: 0.5 });

  return (
    <section
      ref={ref}
      id="slide-6"
      className="h-screen w-full snap-start snap-always flex items-center justify-center relative overflow-hidden"
      style={{ background: "#ffffff" }}
    >
      <AnimatePresence>
        {isInView && (
          <motion.div
            initial="hidden"
            animate="visible"
            className="w-full h-full flex items-center justify-center px-8 md:px-16 lg:px-24"
          >
            <div className="w-full max-w-3xl">
              <motion.p
                variants={fadeUp}
                custom={0}
                style={{
                  fontFamily: "var(--font-ibm-plex-mono)",
                  fontWeight: 300,
                  fontSize: "clamp(0.7rem, 1vw, 0.85rem)",
                  color: MUTED,
                  letterSpacing: "0.2em",
                  textTransform: "uppercase",
                  textAlign: "center",
                  marginBottom: "3rem",
                }}
              >
                Model Comparison &middot; mRMSE &darr;
              </motion.p>
              <motion.div variants={fadeUp} custom={1}>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart
                    data={comparisonData}
                    layout="vertical"
                    margin={{ top: 0, right: 60, bottom: 0, left: 20 }}
                    barCategoryGap="30%"
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      horizontal={false}
                      stroke="#e5e7eb"
                    />
                    <XAxis
                      type="number"
                      domain={[0, 1.4]}
                      tick={{
                        fontFamily: "var(--font-ibm-plex-mono)",
                        fontSize: 11,
                        fill: MUTED,
                      }}
                      axisLine={{ stroke: "#e5e7eb" }}
                      tickLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={120}
                      tick={{
                        fontFamily: "var(--font-ibm-plex-mono)",
                        fontSize: 12,
                        fill: DARK,
                        fontWeight: 300,
                      }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Bar dataKey="mrmse" radius={[0, 4, 4, 0]} barSize={28}>
                      {comparisonData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                      <LabelList
                        dataKey="mrmse"
                        position="right"
                        style={{
                          fontFamily: "var(--font-ibm-plex-mono)",
                          fontSize: 13,
                          fill: DARK,
                          fontWeight: 400,
                        }}
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}

/* ─── Slide 7: Per-molecule Results ─── */
const moleculeResults = [
  { formula: "H\u2082O", name: "Water", rmse: 0.218, color: "#0ea5e9" },
  {
    formula: "CO\u2082",
    name: "Carbon Dioxide",
    rmse: 0.261,
    color: "#f59e0b",
  },
  {
    formula: "CO",
    name: "Carbon Monoxide",
    rmse: 0.327,
    color: "#ef4444",
  },
  { formula: "CH\u2084", name: "Methane", rmse: 0.29, color: "#22c55e" },
  { formula: "NH\u2083", name: "Ammonia", rmse: 0.378, color: "#a855f7" },
];

function SlideMoleculeResults() {
  return (
    <SlideWrapper id={7}>
      <div className="w-full max-w-4xl">
        <motion.p
          variants={fadeUp}
          custom={0}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.7rem, 1vw, 0.85rem)",
            color: MUTED,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            textAlign: "center",
            marginBottom: "3rem",
          }}
        >
          Per-Molecule RMSE
        </motion.p>
        <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-5 gap-4 md:gap-6">
          {moleculeResults.map((mol, i) => (
            <motion.div
              key={mol.formula}
              variants={fadeUp}
              custom={i + 1}
              style={{
                background: "#ffffff",
                border: `1px solid ${LIGHT_GRAY}`,
                borderRadius: "12px",
                padding: "1.5rem 1rem",
                textAlign: "center",
                position: "relative",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  height: "3px",
                  background: mol.color,
                }}
              />
              <div
                style={{
                  fontFamily: "var(--font-outfit)",
                  fontWeight: 400,
                  fontSize: "1.1rem",
                  color: mol.color,
                  marginBottom: "0.25rem",
                }}
              >
                {mol.formula}
              </div>
              <div
                style={{
                  fontFamily: "var(--font-ibm-plex-mono)",
                  fontWeight: 300,
                  fontSize: "0.65rem",
                  color: "#9ca3af",
                  marginBottom: "1rem",
                  letterSpacing: "0.05em",
                }}
              >
                {mol.name}
              </div>
              <div
                style={{
                  fontFamily: "var(--font-ibm-plex-mono)",
                  fontWeight: 500,
                  fontSize: "1.75rem",
                  color: DARK,
                  letterSpacing: "-0.02em",
                }}
              >
                {mol.rmse.toFixed(3)}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </SlideWrapper>
  );
}

/* ─── Slide 8: Efficiency Stats ─── */
const stats = [
  { value: "3", unit: "min", label: "Training Time" },
  { value: "120K", unit: "", label: "Parameters" },
  { value: "12", unit: "qubits", label: "Quantum Circuit" },
];

function SlideEfficiency() {
  return (
    <SlideWrapper id={8}>
      <div className="w-full max-w-4xl">
        <motion.p
          variants={fadeUp}
          custom={0}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.7rem, 1vw, 0.85rem)",
            color: MUTED,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            textAlign: "center",
            marginBottom: "4rem",
          }}
        >
          Efficiency
        </motion.p>
        <div className="flex flex-col md:flex-row items-center justify-center gap-12 md:gap-20">
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              variants={fadeUp}
              custom={i + 1}
              className="text-center"
            >
              <div className="flex items-baseline justify-center gap-2">
                <span
                  style={{
                    fontFamily: "var(--font-outfit)",
                    fontWeight: 300,
                    fontSize: "clamp(3rem, 7vw, 5rem)",
                    color: DARK,
                    letterSpacing: "-0.03em",
                    lineHeight: 1,
                  }}
                >
                  {stat.value}
                </span>
                {stat.unit && (
                  <span
                    style={{
                      fontFamily: "var(--font-ibm-plex-mono)",
                      fontWeight: 300,
                      fontSize: "clamp(1rem, 2vw, 1.5rem)",
                      color: TEAL,
                    }}
                  >
                    {stat.unit}
                  </span>
                )}
              </div>
              <div
                style={{
                  fontFamily: "var(--font-ibm-plex-mono)",
                  fontWeight: 300,
                  fontSize: "0.8rem",
                  color: MUTED,
                  marginTop: "0.75rem",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                }}
              >
                {stat.label}
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div
          variants={fadeUp}
          custom={5}
          className="flex items-center justify-center gap-3 mt-16"
        >
          {["3 min", "120K params", "12 qubits"].map((item, i) => (
            <span key={item} className="flex items-center gap-3">
              <span
                style={{
                  fontFamily: "var(--font-ibm-plex-mono)",
                  fontWeight: 300,
                  fontSize: "clamp(0.75rem, 1vw, 0.9rem)",
                  color: MUTED,
                }}
              >
                {item}
              </span>
              {i < 2 && (
                <span style={{ color: "#d1d5db" }}>&middot;</span>
              )}
            </span>
          ))}
        </motion.div>
      </div>
    </SlideWrapper>
  );
}

/* ─── Slide 9: Closing ─── */
function SlideClosing() {
  return (
    <SlideWrapper id={9}>
      <div className="text-center max-w-4xl">
        <motion.h2
          variants={fadeUp}
          custom={0}
          style={{
            fontFamily: "var(--font-outfit)",
            fontWeight: 300,
            fontSize: "clamp(1.8rem, 4.5vw, 3.5rem)",
            color: DARK,
            letterSpacing: "-0.02em",
            lineHeight: 1.2,
          }}
        >
          First{" "}
          <span style={{ color: TEAL }}>quantum ML</span>
          <br />
          for biosignature detection.
        </motion.h2>
        <motion.div
          variants={fadeUp}
          custom={1}
          style={{
            width: "60px",
            height: "1px",
            background: TEAL,
            margin: "2.5rem auto",
          }}
        />
        <motion.p
          variants={fadeUp}
          custom={2}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "clamp(0.75rem, 1vw, 0.9rem)",
            color: MUTED,
            lineHeight: 1.8,
            maxWidth: "28rem",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        >
          A quantum-classical hybrid that predicts molecular abundances
          from transmission spectra &mdash; faster, lighter, and more
          accurate than classical baselines.
        </motion.p>
        <motion.div
          variants={fadeIn}
          custom={3}
          style={{
            fontFamily: "var(--font-ibm-plex-mono)",
            fontWeight: 300,
            fontSize: "0.75rem",
            color: "#9ca3af",
            letterSpacing: "0.15em",
            textTransform: "uppercase",
            marginTop: "3rem",
          }}
        >
          HACK-4-SAGES 2026
        </motion.div>
      </div>
    </SlideWrapper>
  );
}

/* ─── Dot Navigation ─── */
function DotNav({
  current,
  onDot,
}: {
  current: number;
  onDot: (i: number) => void;
}) {
  return (
    <nav
      className="fixed right-6 top-1/2 -translate-y-1/2 z-50 flex flex-col gap-3"
      aria-label="Slide navigation"
    >
      {Array.from({ length: TOTAL_SLIDES }, (_, i) => (
        <button
          key={i}
          onClick={() => onDot(i)}
          aria-label={`Go to slide ${i + 1}`}
          className="group flex items-center justify-center"
          style={{ width: "20px", height: "20px" }}
        >
          <motion.div
            animate={{
              width: current === i ? 10 : 6,
              height: current === i ? 10 : 6,
              backgroundColor: current === i ? TEAL : "#d1d5db",
            }}
            transition={{ duration: 0.3 }}
            style={{ borderRadius: "50%" }}
          />
        </button>
      ))}
    </nav>
  );
}

/* ─── Slide Counter ─── */
function SlideCounter({ current }: { current: number }) {
  const padded = String(current + 1).padStart(2, "0");
  const total = String(TOTAL_SLIDES).padStart(2, "0");

  return (
    <div
      className="fixed bottom-6 right-6 z-50"
      style={{
        fontFamily: "var(--font-ibm-plex-mono)",
        fontWeight: 300,
        fontSize: "0.75rem",
        color: MUTED,
        letterSpacing: "0.1em",
      }}
    >
      <span style={{ color: DARK }}>{padded}</span>
      <span style={{ margin: "0 0.25rem" }}>/</span>
      <span>{total}</span>
    </div>
  );
}

/* ─── Main Page ─── */
export default function ExoBiomePage() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  const scrollToSlide = useCallback((index: number) => {
    const el = document.getElementById(`slide-${index + 1}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.id;
            const num = parseInt(id.replace("slide-", ""), 10);
            if (!isNaN(num)) {
              setCurrentSlide(num - 1);
            }
          }
        });
      },
      {
        root: container,
        threshold: 0.5,
      }
    );

    const slides = container.querySelectorAll("[id^='slide-']");
    slides.forEach((slide) => observerRef.current?.observe(slide));

    return () => {
      observerRef.current?.disconnect();
    };
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown" || e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault();
        const next = Math.min(currentSlide + 1, TOTAL_SLIDES - 1);
        scrollToSlide(next);
      } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
        e.preventDefault();
        const prev = Math.max(currentSlide - 1, 0);
        scrollToSlide(prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [currentSlide, scrollToSlide]);

  return (
    <div
      className={`${outfit.variable} ${ibmPlexMono.variable}`}
      style={{ background: "#ffffff" }}
    >
      <div
        ref={containerRef}
        className="h-screen overflow-y-auto snap-y snap-mandatory"
        style={{
          scrollBehavior: "smooth",
          WebkitOverflowScrolling: "touch",
        }}
      >
        <SlideTitle />
        <SlideQuestion />
        <SlideMolecules />
        <SlideArchitecture />
        <SlideNumber />
        <SlideComparison />
        <SlideMoleculeResults />
        <SlideEfficiency />
        <SlideClosing />
      </div>

      <DotNav current={currentSlide} onDot={scrollToSlide} />
      <SlideCounter current={currentSlide} />
    </div>
  );
}
