import { ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer className="mt-12">
      {/* Separator */}
      <hr className="h-px border-0 bg-gradient-to-r from-transparent via-cyan/30 to-transparent" />

      <div className="py-20">
        <div className="grid gap-12 lg:grid-cols-[2fr_1fr_1fr]">
          {/* About */}
          <div>
            <h4 className="font-display text-lg font-semibold text-heading">
              About ExoBiome
            </h4>
            <p className="mt-3 max-w-md text-sm leading-relaxed text-muted">
              ExoBiome is a research project exploring whether quantum machine
              learning can detect biosignatures in exoplanet atmospheres. Built
              during HACK-4-SAGES 2026 (ETH Zurich COPL), it combines real
              quantum hardware with spectral analysis to push the frontier of
              life detection beyond Earth.
            </p>
          </div>

          {/* Research */}
          <div>
            <h4 className="font-display text-lg font-semibold text-heading">
              Research
            </h4>
            <ul className="mt-3 space-y-3 text-sm">
              <li>
                <a
                  href="https://arxiv.org/abs/2509.03617"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-muted transition-colors hover:text-cyan"
                >
                  Vetrano et al. 2025 — QELM
                  <ExternalLink size={12} />
                </a>
              </li>
              <li>
                <a
                  href="https://doi.org/10.1093/mnras/stae2948"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-muted transition-colors hover:text-cyan"
                >
                  Cardenas et al. 2025 — MultiREx
                  <ExternalLink size={12} />
                </a>
              </li>
            </ul>
          </div>

          {/* Team */}
          <div>
            <h4 className="font-display text-lg font-semibold text-heading">
              Team
            </h4>
            <ul className="mt-3 space-y-1.5 text-sm text-muted">
              <li>Michal Szczesny</li>
              <li>Team Member 2</li>
              <li>Team Member 3</li>
              <li>Team Member 4</li>
            </ul>
            <p className="mt-4 text-xs text-muted/60">HACK-4-SAGES 2026</p>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-12 flex flex-wrap items-center gap-3 rounded-xl bg-surface/30 px-6 py-3">
          <span className="text-xs font-medium text-muted">Built with</span>
          {["Next.js", "React", "Tailwind", "Nivo", "Qiskit", "sQUlearn"].map(
            (tech) => (
              <span
                key={tech}
                className="rounded-full border border-border px-2.5 py-0.5 text-xs text-muted"
              >
                {tech}
              </span>
            )
          )}
        </div>
      </div>
    </footer>
  );
}
