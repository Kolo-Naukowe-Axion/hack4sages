"use client";

import { useScrollReveal } from "@/hooks/useScrollReveal";
import { GradientMesh } from "@/components/GradientMesh";
import { Hero } from "@/components/Hero";
import { Stats } from "@/components/Stats";
import { Features } from "@/components/Features";
import { HowItWorks } from "@/components/HowItWorks";
import { CodeSnippet } from "@/components/CodeSnippet";
import { CtaSection } from "@/components/CtaSection";
import { Footer } from "@/components/Footer";

export default function Home() {
  const ref = useScrollReveal();

  return (
    <div ref={ref}>
      <GradientMesh />
      <Hero />
      <Stats />
      <Features />
      <HowItWorks />
      <CodeSnippet />
      <CtaSection />
      <Footer />
    </div>
  );
}
