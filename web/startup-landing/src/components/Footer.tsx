import { Atom } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-border py-10 px-6">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-muted text-xs">
          <Atom className="w-4 h-4" />
          <span>ExoBiome</span>
          <span className="mx-1">·</span>
          <span>Team Axion</span>
        </div>
        <div className="text-xs text-muted/60">
          HACK-4-SAGES 2026 · ETH Zurich COPL · Life Detection &amp; Biosignatures
        </div>
      </div>
    </footer>
  );
}
