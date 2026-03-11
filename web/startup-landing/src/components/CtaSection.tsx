import Link from "next/link";
import { ArrowRight } from "lucide-react";

export function CtaSection() {
  return (
    <section className="relative py-32 px-6">
      <div className="max-w-3xl mx-auto text-center">
        <h2 className="reveal text-3xl sm:text-5xl font-bold mb-4">
          Ready to <span className="gradient-text">explore?</span>
        </h2>
        <p className="reveal reveal-delay-1 text-muted mb-10 max-w-lg mx-auto">
          Browse 16 exoplanets, run quantum-powered biosignature detection, and compare
          model performance in real time.
        </p>
        <div className="reveal reveal-delay-2">
          <Link
            href="/explorer"
            className="gradient-bg inline-flex items-center gap-2 px-8 py-3.5 text-sm font-semibold rounded-xl text-white hover:opacity-90 transition-opacity"
          >
            Launch Explorer
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </div>
    </section>
  );
}
