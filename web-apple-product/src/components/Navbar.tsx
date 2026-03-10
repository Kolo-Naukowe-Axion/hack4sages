"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Menu, X } from "lucide-react";

const links = [
  { href: "/", label: "Home" },
  { href: "/explorer", label: "Explorer" },
  { href: "/models", label: "Models" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav aria-label="Main navigation" className="fixed top-0 left-0 right-0 z-50 h-16 border-b border-border/60 bg-void/80 backdrop-blur-xl backdrop-saturate-150">
      <div className="mx-auto flex h-full max-w-7xl items-center justify-between px-6 lg:px-8">
        <Link
          href="/"
          className="flex items-center gap-0.5 font-display text-xl font-semibold tracking-tight text-heading"
        >
          Exo
          <span className="relative">
            B
            <span className="absolute -top-0.5 right-[-2px] h-1.5 w-1.5 rounded-full bg-red" />
          </span>
          iome
        </Link>

        <div className="hidden items-center gap-1 md:flex">
          {links.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`relative px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-heading/20 focus-visible:rounded-lg ${
                  active
                    ? "text-heading"
                    : "text-muted hover:text-heading"
                }`}
              >
                {label}
                {active && (
                  <span className="absolute bottom-0 left-3 right-3 h-0.5 rounded-full bg-heading" />
                )}
              </Link>
            );
          })}
        </div>

        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="flex items-center justify-center rounded-lg p-2 text-muted transition-colors hover:text-heading focus:outline-none focus-visible:ring-2 focus-visible:ring-heading/20 md:hidden"
          aria-label="Toggle menu"
          aria-expanded={mobileOpen}
        >
          <span>{mobileOpen ? <X size={20} /> : <Menu size={20} />}</span>
        </button>
      </div>

      {mobileOpen && (
        <div className="border-b border-border/60 bg-void/95 backdrop-blur-xl md:hidden">
          <div className="flex flex-col gap-1 px-6 py-4">
            {links.map(({ href, label }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={() => setMobileOpen(false)}
                  className={`rounded-xl px-4 py-3 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-heading/20 ${
                    active
                      ? "bg-surface text-heading"
                      : "text-muted hover:bg-surface hover:text-heading"
                  }`}
                >
                  {label}
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </nav>
  );
}
