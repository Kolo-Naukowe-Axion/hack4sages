import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ExoBiome | Scientific Presentation",
  description:
    "A science-first presentation app for ExoBiome: quantum biosignature detection from transmission spectroscopy.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
