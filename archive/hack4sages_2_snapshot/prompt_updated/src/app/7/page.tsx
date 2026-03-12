"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Outfit } from "next/font/google";

const outfit = Outfit({
  subsets: ["latin"],
  weight: ["300", "400"],
  variable: "--font-outfit",
});

const TOTAL_SLIDES = 12;
const GOLD = "#d4a574";

function SlideCounter({ current }: { current: number }) {
  return (
    <div
      className="fixed bottom-8 right-10 z-50 font-mono text-xs tracking-widest"
      style={{ color: "rgba(255,255,255,0.3)" }}
    >
      {String(current + 1).padStart(2, "0")}/{TOTAL_SLIDES}
    </div>
  );
}

function FadeUp({
  children,
  delay = 0,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.8, delay, ease: [0.25, 0.1, 0.25, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// Slide 0: Just the name
function Slide0() {
  return (
    <div className="flex items-center justify-center h-full w-full">
      <FadeUp>
        <h1
          className="text-[8vw] font-light tracking-[-0.04em] text-white leading-none"
          style={{ fontFamily: "var(--font-outfit)" }}
        >
          ExoBiome
        </h1>
      </FadeUp>
    </div>
  );
}

// Slide 1: Subtitle only
function Slide1() {
  return (
    <div className="flex items-center justify-center h-full w-full">
      <FadeUp>
        <p
          className="text-[2.5vw] font-light tracking-[-0.02em] text-white/80 leading-relaxed text-center"
          style={{ fontFamily: "var(--font-outfit)" }}
        >
          Quantum Biosignature Detection
        </p>
      </FadeUp>
    </div>
  );
}

// Slide 2: "5 molecules." then formulas fade in
function Slide2() {
  const molecules = ["H₂O", "CO₂", "CO", "CH₄", "NH₃"];
  return (
    <div className="flex flex-col items-center justify-center h-full w-full gap-16">
      <FadeUp>
        <p
          className="text-[4vw] font-light text-white tracking-[-0.03em]"
          style={{ fontFamily: "var(--font-outfit)" }}
        >
          5 molecules.
        </p>
      </FadeUp>
      <div className="flex gap-8">
        {molecules.map((mol, i) => (
          <FadeUp key={mol} delay={0.6 + i * 0.25}>
            <span
              className="font-mono text-[1.8vw] tracking-wider"
              style={{ color: GOLD }}
            >
              {mol}
            </span>
          </FadeUp>
        ))}
      </div>
    </div>
  );
}

// Slide 3: Molecule line
function Slide3() {
  return (
    <div className="flex items-center justify-center h-full w-full">
      <FadeUp>
        <p
          className="font-mono text-[2vw] tracking-[0.15em]"
          style={{ color: GOLD }}
        >
          H₂O &middot; CO₂ &middot; CO &middot; CH₄ &middot; NH₃
        </p>
      </FadeUp>
    </div>
  );
}

// Slide 4: Architecture — minimal flow
function Slide4() {
  const steps = [
    { label: "Spectrum", sub: "52 wavelengths" },
    { label: "SpectralEncoder", sub: "1D-CNN" },
    { label: "Fusion", sub: "128-dim" },
    { label: "Quantum Circuit", sub: "12 qubits" },
    { label: "5 molecules", sub: "log₁₀ VMR" },
  ];

  return (
    <div className="flex items-center justify-center h-full w-full px-12">
      <div className="flex items-center gap-0 w-full max-w-5xl justify-between">
        {steps.map((step, i) => (
          <FadeUp key={step.label} delay={i * 0.2} className="flex items-center">
            <div className="flex flex-col items-center text-center min-w-[120px]">
              <span
                className="text-sm font-light tracking-wide text-white/90"
                style={{ fontFamily: "var(--font-outfit)" }}
              >
                {step.label}
              </span>
              <span className="text-xs font-mono mt-1" style={{ color: GOLD }}>
                {step.sub}
              </span>
            </div>
            {i < steps.length - 1 && (
              <div className="mx-4 flex items-center">
                <div
                  className="w-12 h-px"
                  style={{ backgroundColor: "rgba(255,255,255,0.15)" }}
                />
                <div
                  className="w-0 h-0 border-t-[3px] border-b-[3px] border-l-[5px] border-transparent"
                  style={{ borderLeftColor: "rgba(255,255,255,0.15)" }}
                />
              </div>
            )}
          </FadeUp>
        ))}
      </div>
    </div>
  );
}

// Slide 5: "0.295" — the number, enormous, gold
function Slide5() {
  return (
    <div className="flex items-center justify-center h-full w-full">
      <FadeUp>
        <span
          className="font-mono text-[14vw] font-light leading-none tracking-[-0.04em]"
          style={{ color: GOLD }}
        >
          0.295
        </span>
      </FadeUp>
    </div>
  );
}

// Slide 6: mRMSE explanation + comparison bars
function Slide6() {
  const models = [
    { name: "Random Forest", score: 1.2, width: "100%" },
    { name: "CNN Baseline", score: 0.85, width: "70.8%" },
    { name: "ADC 2023 Winner", score: 0.32, width: "26.7%" },
    { name: "ExoBiome", score: 0.295, width: "24.6%", highlight: true },
  ];

  return (
    <div className="flex flex-col items-center justify-center h-full w-full px-12">
      <FadeUp>
        <p
          className="text-lg font-light tracking-widest uppercase text-white/40 mb-16"
          style={{ fontFamily: "var(--font-outfit)" }}
        >
          mean Relative RMSE
        </p>
      </FadeUp>
      <div className="w-full max-w-2xl space-y-6">
        {models.map((m, i) => (
          <FadeUp key={m.name} delay={0.3 + i * 0.15}>
            <div className="flex items-center gap-6">
              <span
                className="text-sm font-light w-40 text-right shrink-0"
                style={{
                  fontFamily: "var(--font-outfit)",
                  color: m.highlight ? GOLD : "rgba(255,255,255,0.5)",
                }}
              >
                {m.name}
              </span>
              <div className="flex-1 relative h-7 flex items-center">
                <motion.div
                  className="h-[2px] rounded-full"
                  style={{
                    backgroundColor: m.highlight ? GOLD : "rgba(255,255,255,0.15)",
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: m.width }}
                  transition={{
                    duration: 1,
                    delay: 0.5 + i * 0.15,
                    ease: [0.25, 0.1, 0.25, 1],
                  }}
                />
                <motion.span
                  className="absolute font-mono text-xs"
                  style={{
                    left: m.width,
                    color: m.highlight ? GOLD : "rgba(255,255,255,0.35)",
                    paddingLeft: 8,
                  }}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 1.2 + i * 0.15 }}
                >
                  {m.score}
                </motion.span>
              </div>
            </div>
          </FadeUp>
        ))}
      </div>
    </div>
  );
}

// Slide 7: Per-molecule mini bars
function Slide7() {
  const molecules = [
    { name: "H₂O", score: 0.218 },
    { name: "CO₂", score: 0.261 },
    { name: "CH₄", score: 0.29 },
    { name: "CO", score: 0.327 },
    { name: "NH₃", score: 0.378 },
  ];

  const maxScore = 0.5;

  return (
    <div className="flex flex-col items-center justify-center h-full w-full px-12">
      <FadeUp>
        <p
          className="text-lg font-light tracking-widest uppercase text-white/40 mb-16"
          style={{ fontFamily: "var(--font-outfit)" }}
        >
          Per-molecule mRMSE
        </p>
      </FadeUp>
      <div className="w-full max-w-lg space-y-8">
        {molecules.map((m, i) => (
          <FadeUp key={m.name} delay={0.2 + i * 0.12}>
            <div className="flex items-center gap-6">
              <span
                className="font-mono text-sm w-12 text-right shrink-0"
                style={{ color: GOLD }}
              >
                {m.name}
              </span>
              <div className="flex-1 relative h-6 flex items-center">
                <motion.div
                  className="h-[2px] rounded-full"
                  style={{ backgroundColor: GOLD }}
                  initial={{ width: 0 }}
                  animate={{
                    width: `${(m.score / maxScore) * 100}%`,
                  }}
                  transition={{
                    duration: 0.9,
                    delay: 0.5 + i * 0.12,
                    ease: [0.25, 0.1, 0.25, 1],
                  }}
                />
                <motion.span
                  className="absolute font-mono text-xs text-white/40"
                  style={{
                    left: `${(m.score / maxScore) * 100}%`,
                    paddingLeft: 8,
                  }}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 1 + i * 0.12 }}
                >
                  {m.score}
                </motion.span>
              </div>
            </div>
          </FadeUp>
        ))}
      </div>
    </div>
  );
}

// Slide 8: "3 minutes to train."
function Slide8() {
  return (
    <div className="flex items-center justify-center h-full w-full">
      <FadeUp>
        <p
          className="text-[4vw] font-light text-white tracking-[-0.03em] text-center"
          style={{ fontFamily: "var(--font-outfit)" }}
        >
          3 minutes to train.
        </p>
      </FadeUp>
    </div>
  );
}

// Slide 9: "120K parameters."
function Slide9() {
  return (
    <div className="flex items-center justify-center h-full w-full">
      <FadeUp>
        <div className="text-center">
          <span
            className="font-mono text-[6vw] tracking-[-0.03em]"
            style={{ color: GOLD }}
          >
            120K
          </span>
          <p
            className="text-xl font-light text-white/40 mt-4 tracking-wide"
            style={{ fontFamily: "var(--font-outfit)" }}
          >
            parameters
          </p>
        </div>
      </FadeUp>
    </div>
  );
}

// Slide 10: "12 qubits."
function Slide10() {
  return (
    <div className="flex flex-col items-center justify-center h-full w-full gap-10">
      <FadeUp>
        <span
          className="font-mono text-[10vw] tracking-[-0.04em] leading-none"
          style={{ color: GOLD }}
        >
          12
        </span>
      </FadeUp>
      <FadeUp delay={0.4}>
        <p
          className="text-2xl font-light text-white/40 tracking-widest uppercase"
          style={{ fontFamily: "var(--font-outfit)" }}
        >
          qubits
        </p>
      </FadeUp>
    </div>
  );
}

// Slide 11: Closing statement
function Slide11() {
  return (
    <div className="flex flex-col items-center justify-center h-full w-full px-12 gap-12">
      <FadeUp>
        <p
          className="text-[3vw] font-light text-white text-center leading-snug tracking-[-0.02em] max-w-4xl"
          style={{ fontFamily: "var(--font-outfit)" }}
        >
          First quantum ML applied to
          <br />
          <span style={{ color: GOLD }}>biosignature detection.</span>
        </p>
      </FadeUp>
      <FadeUp delay={0.6}>
        <p
          className="text-sm font-light tracking-[0.2em] uppercase text-white/20"
          style={{ fontFamily: "var(--font-outfit)" }}
        >
          HACK-4-SAGES 2026 &middot; ETH Zurich
        </p>
      </FadeUp>
    </div>
  );
}

const SLIDES = [
  Slide0,
  Slide1,
  Slide2,
  Slide3,
  Slide4,
  Slide5,
  Slide6,
  Slide7,
  Slide8,
  Slide9,
  Slide10,
  Slide11,
];

export default function Page() {
  const [current, setCurrent] = useState(0);
  const isScrolling = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const goTo = useCallback(
    (index: number) => {
      if (index < 0 || index >= TOTAL_SLIDES || isScrolling.current) return;
      isScrolling.current = true;
      setCurrent(index);
      setTimeout(() => {
        isScrolling.current = false;
      }, 900);
    },
    []
  );

  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (isScrolling.current) return;
      if (e.deltaY > 0) goTo(current + 1);
      else if (e.deltaY < 0) goTo(current - 1);
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown" || e.key === " " || e.key === "PageDown") {
        e.preventDefault();
        goTo(current + 1);
      } else if (e.key === "ArrowUp" || e.key === "PageUp") {
        e.preventDefault();
        goTo(current - 1);
      }
    };

    let touchStartY = 0;
    const handleTouchStart = (e: TouchEvent) => {
      touchStartY = e.touches[0].clientY;
    };
    const handleTouchEnd = (e: TouchEvent) => {
      const delta = touchStartY - e.changedTouches[0].clientY;
      if (Math.abs(delta) > 50) {
        if (delta > 0) goTo(current + 1);
        else goTo(current - 1);
      }
    };

    const el = containerRef.current;
    if (!el) return;

    el.addEventListener("wheel", handleWheel, { passive: false });
    window.addEventListener("keydown", handleKeyDown);
    el.addEventListener("touchstart", handleTouchStart, { passive: true });
    el.addEventListener("touchend", handleTouchEnd, { passive: true });

    return () => {
      el.removeEventListener("wheel", handleWheel);
      window.removeEventListener("keydown", handleKeyDown);
      el.removeEventListener("touchstart", handleTouchStart);
      el.removeEventListener("touchend", handleTouchEnd);
    };
  }, [current, goTo]);

  const SlideComponent = SLIDES[current];

  return (
    <div
      ref={containerRef}
      className={`${outfit.variable} h-screen w-screen overflow-hidden bg-black relative select-none`}
      style={{ cursor: "default" }}
    >
      {/* Subtle dot navigation on left */}
      <div className="fixed left-8 top-1/2 -translate-y-1/2 z-50 flex flex-col gap-2.5">
        {Array.from({ length: TOTAL_SLIDES }).map((_, i) => (
          <button
            key={i}
            onClick={() => goTo(i)}
            className="w-1.5 h-1.5 rounded-full transition-all duration-500"
            style={{
              backgroundColor:
                i === current ? GOLD : "rgba(255,255,255,0.12)",
              transform: i === current ? "scale(1.4)" : "scale(1)",
            }}
            aria-label={`Go to slide ${i + 1}`}
          />
        ))}
      </div>

      {/* Slide counter */}
      <SlideCounter current={current} />

      {/* Slide content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={current}
          className="h-full w-full"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4 }}
        >
          <SlideComponent />
        </motion.div>
      </AnimatePresence>

      {/* Scroll hint on first slide */}
      {current === 0 && (
        <motion.div
          className="fixed bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 1 }}
        >
          <motion.div
            className="w-px h-8"
            style={{ backgroundColor: "rgba(255,255,255,0.15)" }}
            animate={{ scaleY: [1, 0.5, 1] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          />
          <span
            className="text-[10px] tracking-[0.3em] uppercase text-white/15 font-light"
            style={{ fontFamily: "var(--font-outfit)" }}
          >
            scroll
          </span>
        </motion.div>
      )}
    </div>
  );
}
