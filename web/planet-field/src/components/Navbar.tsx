"use client";

import { Globe, Telescope, Cpu } from "lucide-react";
import { NavBar } from "@/components/ui/tubelight-navbar";

const navItems = [
  { name: "Home", url: "/", icon: Globe },
  { name: "Explorer", url: "/explorer", icon: Telescope },
  { name: "Models", url: "/models", icon: Cpu },
];

export function Navbar() {
  return <NavBar items={navItems} />;
}
