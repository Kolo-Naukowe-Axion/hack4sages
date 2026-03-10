import { ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer className="mt-10 border-t border-border pt-10 pb-16">
      {/* References */}
      <div>
        <h3 className="font-serif text-lg font-semibold text-heading">References</h3>
        <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-muted">
          <li>
            Vetrano, V. et al. (2025). Quantum extreme learning machines for atmospheric retrieval.{" "}
            <a
              href="https://arxiv.org/abs/2509.03617"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-baseline gap-1 text-accent hover:text-accent-light"
            >
              arXiv:2509.03617
              <ExternalLink size={10} />
            </a>
          </li>
          <li>
            Cardenas, R. et al. (2025). MultiREx: Multi-epoch retrieval of exoplanet atmospheres.{" "}
            <a
              href="https://doi.org/10.1093/mnras/stae2948"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-baseline gap-1 text-accent hover:text-accent-light"
            >
              MNRAS, stae2948
              <ExternalLink size={10} />
            </a>
          </li>
        </ol>
      </div>

      <hr className="journal-rule my-8" />

      {/* Acknowledgments & Team */}
      <div className="grid gap-8 text-sm lg:grid-cols-[2fr_1fr]">
        <div>
          <h4 className="font-serif font-semibold text-heading">Acknowledgments</h4>
          <p className="mt-2 text-muted">
            ExoBiome was developed during HACK-4-SAGES 2026, organized by the
            Centre for Orbital Propulsion Laboratory (COPL), ETH Zurich. This work
            explores the application of quantum machine learning to biosignature
            detection in exoplanet atmospheres using real quantum hardware.
          </p>
        </div>
        <div>
          <h4 className="font-serif font-semibold text-heading">Authors</h4>
          <ul className="mt-2 space-y-1 text-muted">
            <li>Michal Szczesny</li>
            <li>Ariel Smolenski</li>
            <li>Mateusz Karandys</li>
            <li>Kacper Kramarz</li>
          </ul>
        </div>
      </div>

      <div className="mt-8 flex flex-wrap items-center gap-2.5">
        <span className="font-sans text-xs font-medium text-muted">Built with</span>
        {["Next.js", "React", "Tailwind", "Nivo", "Qiskit", "sQUlearn"].map(
          (tech) => (
            <span
              key={tech}
              className="border border-border-light px-2 py-0.5 font-sans text-xs text-muted"
            >
              {tech}
            </span>
          )
        )}
      </div>
    </footer>
  );
}
