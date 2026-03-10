"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function NotebookToolbar() {
  const pathname = usePathname();

  const navTabs = [
    { label: "ExoBiome_Analysis.ipynb", href: "/" },
    { label: "Planet_Explorer.ipynb", href: "/explorer" },
    { label: "Model_Comparison.ipynb", href: "/models" },
  ];

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-sm border-b border-nb-border">
      {/* Title bar */}
      <div className="flex items-center px-5 h-[36px] border-b border-nb-border/60">
        <span className="font-mono text-[13px] text-nb-prompt font-medium tracking-tight">
          ExoBiome
        </span>
        <span className="font-mono text-[13px] text-nb-muted ml-2">
          Quantum Biosignature Detection
        </span>
      </div>

      {/* Notebook tabs */}
      <div className="flex items-center px-3 h-[34px] text-[12px] gap-0">
        {navTabs.map((tab) => {
          const active = pathname === tab.href;
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={`px-3 py-1.5 font-mono rounded-t transition-colors ${
                active
                  ? "bg-white text-nb-prompt font-medium border-t-2 border-t-nb-prompt border-x border-nb-border"
                  : "text-nb-muted hover:text-nb-text hover:bg-white/50"
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
