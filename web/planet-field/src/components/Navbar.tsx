"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Orbit, Telescope, Cpu } from "lucide-react";

const links = [
  { href: "/", label: "Home", icon: Orbit },
  { href: "/explorer", label: "Explorer", icon: Telescope },
  { href: "/models", label: "Models", icon: Cpu },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-space-900/80 border-b border-white/5">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan to-green flex items-center justify-center">
            <Orbit className="w-4 h-4 text-space-900" />
          </div>
          <span className="font-bold text-lg text-white tracking-tight">
            ExoBiome
          </span>
        </Link>

        <div className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  active
                    ? "bg-white/10 text-cyan"
                    : "text-white/50 hover:text-white/80 hover:bg-white/5"
                }`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
