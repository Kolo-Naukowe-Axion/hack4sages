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
    <nav className="fixed top-0 left-0 right-0 z-50 bg-cream/95 backdrop-blur-sm">
      <div className="mx-auto flex h-14 max-w-4xl items-center justify-between px-6">
        <Link
          href="/"
          className="font-serif text-lg font-semibold tracking-wide text-heading"
        >
          <span className="text-accent">Exo</span>Biome
        </Link>

        <div className="hidden items-center gap-8 md:flex">
          {links.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`relative font-sans text-sm tracking-wide transition-colors ${
                  active
                    ? "font-medium text-accent"
                    : "text-muted hover:text-heading"
                }`}
              >
                {label}
                <span
                  className={`absolute -bottom-1 left-0 right-0 h-px bg-accent transition-opacity ${
                    active ? "opacity-100" : "opacity-0"
                  }`}
                />
              </Link>
            );
          })}
        </div>

        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="flex items-center justify-center p-2 text-muted transition-colors hover:text-heading md:hidden"
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X size={18} /> : <Menu size={18} />}
        </button>
      </div>

      <div className="journal-rule-double mx-auto max-w-4xl" />

      {mobileOpen && (
        <div className="border-b border-border bg-cream/98 backdrop-blur-sm md:hidden">
          <div className="mx-auto flex max-w-4xl flex-col gap-1 px-6 py-3">
            {links.map(({ href, label }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={() => setMobileOpen(false)}
                  className={`rounded px-3 py-2 font-sans text-sm transition-colors ${
                    active
                      ? "font-medium text-accent"
                      : "text-muted hover:text-heading"
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
