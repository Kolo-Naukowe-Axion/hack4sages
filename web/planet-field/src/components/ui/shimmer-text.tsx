"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface ShimmerTextProps {
  children: React.ReactNode;
  className?: string;
  duration?: number;
  delay?: number;
}

export function ShimmerText({
  children,
  className,
  duration = 1.5,
  delay = 1.5,
}: ShimmerTextProps) {
  return (
    <div className="overflow-hidden">
      <motion.div
        className={cn("inline-block", className)}
        style={{
          WebkitTextFillColor: "transparent",
          background:
            "currentColor linear-gradient(to right, currentColor 0%, rgba(255,255,255,0.6) 40%, rgba(255,255,255,0.6) 60%, currentColor 100%)",
          WebkitBackgroundClip: "text",
          backgroundClip: "text",
          backgroundRepeat: "no-repeat",
          backgroundSize: "50% 200%",
        } as React.CSSProperties}
        initial={{
          backgroundPositionX: "250%",
        }}
        animate={{
          backgroundPositionX: ["-100%", "250%"],
        }}
        transition={{
          duration: duration,
          delay: delay,
          repeat: Infinity,
          repeatDelay: 1.5,
          ease: "linear",
        }}
      >
        <span>{children}</span>
      </motion.div>
    </div>
  );
}
