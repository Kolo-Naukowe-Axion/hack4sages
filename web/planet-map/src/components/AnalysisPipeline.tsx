"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { QuantumCircuit } from "./QuantumCircuit";
import { Sliders, Atom, Cpu, Brain, Check } from "lucide-react";

interface Props {
  active: boolean;
  onComplete: () => void;
}

const STAGES = [
  { icon: Sliders, label: "Extracting spectral features", sub: "PCA reduction: 50 \u2192 5 components", duration: 800 },
  { icon: Atom, label: "Encoding into quantum state", sub: "Angle encoding: RY rotations on 5 qubits", duration: 600 },
  { icon: Cpu, label: "Executing on IQM Spark", sub: "Random reservoir + entangling layer", duration: 3600 },
  { icon: Brain, label: "Classifying biosignatures", sub: "Ridge regression on measurement readout", duration: 800 },
];

export function AnalysisPipeline({ active, onComplete }: Props) {
  const [currentStage, setCurrentStage] = useState(-1);
  const [circuitDone, setCircuitDone] = useState(false);

  useEffect(() => {
    if (!active) {
      setCurrentStage(-1);
      setCircuitDone(false);
      return;
    }
    setCurrentStage(0);
  }, [active]);

  useEffect(() => {
    if (currentStage >= STAGES.length && active) {
      onComplete();
    }
  }, [currentStage, active, onComplete]);

  useEffect(() => {
    if (!active || currentStage < 0 || currentStage >= STAGES.length) return;
    if (currentStage === 2) return;

    let mounted = true;
    const timer = setTimeout(() => {
      if (mounted) setCurrentStage((s) => s + 1);
    }, STAGES[currentStage].duration);

    return () => {
      mounted = false;
      clearTimeout(timer);
    };
  }, [active, currentStage]);

  useEffect(() => {
    if (active && currentStage === 2 && circuitDone) {
      setCurrentStage(3);
    }
  }, [active, currentStage, circuitDone]);

  const handleCircuitComplete = useCallback(() => {
    setCircuitDone(true);
  }, []);

  if (!active) return null;

  return (
    <div className="bg-surface border border-border rounded-lg p-6">
      <div className="space-y-3">
        {STAGES.map((stage, i) => {
          const Icon = stage.icon;
          const isActive = currentStage === i;
          const isDone = currentStage > i;
          const isPending = currentStage < i;

          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: isPending ? 0.3 : 1, x: 0 }}
              transition={{ delay: i * 0.1, duration: 0.3 }}
              className="flex items-start gap-3"
            >
              <div className={`mt-0.5 w-6 h-6 rounded-md flex items-center justify-center shrink-0 ${
                isDone
                  ? "bg-accent-green/10 text-accent-green"
                  : isActive
                  ? "bg-accent-blue/10 text-accent-blue"
                  : "bg-surface text-muted"
              }`}>
                {isDone ? (
                  <Check className="w-3.5 h-3.5" />
                ) : isActive ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Icon className="w-3.5 h-3.5" />
                  </motion.div>
                ) : (
                  <Icon className="w-3.5 h-3.5" />
                )}
              </div>
              <div className="min-w-0">
                <div className={`text-sm font-medium ${
                  isDone ? "text-accent-green" : isActive ? "text-text" : "text-muted"
                }`}>
                  {stage.label}
                </div>
                <div className="text-[11px] text-muted">{stage.sub}</div>
              </div>
            </motion.div>
          );
        })}
      </div>

      <AnimatePresence>
        {currentStage === 2 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mt-4 pt-4 border-t border-border"
          >
            <QuantumCircuit onComplete={handleCircuitComplete} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
