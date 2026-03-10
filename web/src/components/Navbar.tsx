"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Home, Telescope, Cpu, Menu, X } from "lucide-react";

const links = [
  { href: "/", label: "Home", icon: Home },
  { href: "/explorer", label: "Explorer", icon: Telescope },
  { href: "/models", label: "Models", icon: Cpu },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-16 border-b border-border bg-void/80 backdrop-blur-xl">
      <div className="mx-auto flex h-full max-w-7xl items-center justify-between px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-1 font-display text-xl font-bold text-heading">
          Exo
          <span className="relative">
            B
            <span className="absolute -top-0.5 right-[-2px] h-1.5 w-1.5 rounded-full bg-cyan" />
          </span>
          iome
        </Link>

        <div className="hidden items-center gap-1 md:flex">
          {links.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`relative flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-cyan/30 ${
                  active
                    ? "text-cyan"
                    : "text-muted hover:text-heading"
                }`}
              >
                <Icon size={16} />
                {label}
                {active && (
                  <span className="absolute bottom-0 left-2 right-2 h-0.5 rounded-full bg-cyan shadow-[0_0_8px_var(--color-cyan)]" />
                )}
              </Link>
            );
          })}
        </div>

        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="flex items-center justify-center rounded-lg p-2 text-muted transition-colors hover:text-heading focus:outline-none focus:ring-2 focus:ring-cyan/30 md:hidden"
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {mobileOpen && (
        <div className="border-b border-border bg-void/95 backdrop-blur-xl md:hidden">
          <div className="flex flex-col gap-1 px-6 py-4">
            {links.map(({ href, label, icon: Icon }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={() => setMobileOpen(false)}
                  className={`flex items-center gap-3 rounded-lg px-4 py-3 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-cyan/30 ${
                    active
                      ? "bg-cyan/5 text-cyan"
                      : "text-muted hover:bg-surface/50 hover:text-heading"
                  }`}
                >
                  <Icon size={16} />
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
