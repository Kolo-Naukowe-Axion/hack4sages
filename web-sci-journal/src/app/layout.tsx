import type { Metadata } from "next";
import { Crimson_Pro, Source_Serif_4, Source_Sans_3, JetBrains_Mono } from "next/font/google";
import Navbar from "@/components/Navbar";
import "./globals.css";

const crimsonPro = Crimson_Pro({
  variable: "--font-crimson-pro",
  subsets: ["latin"],
  display: "swap",
});

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
  display: "swap",
});

const sourceSans = Source_Sans_3({
  variable: "--font-source-sans",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "ExoBiome — Quantum Biosignature Detection",
  description:
    "Quantum-enhanced detection of atmospheric biosignatures in exoplanetary spectra",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${crimsonPro.variable} ${sourceSerif.variable} ${sourceSans.variable} ${jetbrainsMono.variable} antialiased`}
      >
        <Navbar />
        <main className="pt-14">{children}</main>
      </body>
    </html>
  );
}
