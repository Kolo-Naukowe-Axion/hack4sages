import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import NotebookToolbar from "@/components/NotebookToolbar";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
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
    "Quantum biosignature detection in exoplanet atmospheres",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} antialiased`}
      >
        <NotebookToolbar />
        <main className="pt-[74px] pb-16">{children}</main>
      </body>
    </html>
  );
}
