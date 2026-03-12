import type { Metadata } from "next";
import "./globals.css";
import { DemoNav } from "@/app/components/DemoNav";

export const metadata: Metadata = {
  title: "ExoBiome Presentation Lab",
  description:
    "Eight design variants for presenting quantum biosignature detection on Ariel transmission spectroscopy data."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <div className="shell topbar-inner">
            <div className="brand">
              <span className="eyebrow">HACK-4-SAGES 2026 • ETH Zurich orbit</span>
              <strong>ExoBiome Presentation Lab</strong>
            </div>
            <DemoNav />
          </div>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
