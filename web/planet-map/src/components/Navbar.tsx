"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Globe, Telescope, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { name: "Home", href: "/", icon: Globe },
  { name: "Explorer", href: "/explorer", icon: Telescope },
  { name: "Models", href: "/models", icon: Cpu },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border bg-bg/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link href="/" className="text-sm font-semibold text-text tracking-tight">
          ExoBiome
        </Link>
        <div className="flex items-center gap-1">
          {links.map(({ name, href, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={name}
                href={href}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm transition-colors",
                  active
                    ? "bg-elevated text-text"
                    : "text-muted hover:text-text hover:bg-elevated/50"
                )}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden sm:inline">{name}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
