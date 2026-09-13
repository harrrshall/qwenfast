import type { Metadata, Viewport } from "next";
import { Newsreader, IBM_Plex_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import "./globals.css";

// Only the weights the stylesheet actually uses: 400 body, 600 headings/strong,
// italic for user turns. Trimmed from 300/400/500/600 x normal/italic (8 files)
// to 4, which halves the font bytes fetched on first paint. `display: swap` plus
// next/font's generated size-adjusted fallback keeps CLS at zero.
const serif = Newsreader({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-serif",
  weight: ["400", "600"],
  style: ["normal", "italic"],
  preload: true,
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono",
  weight: ["400", "500"],
  preload: true,
});

export const metadata: Metadata = {
  title: "qwenfast",
  description: "World's fastest inference for Qwen3.8-27B",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fbfaf7" },
    { media: "(prefers-color-scheme: dark)", color: "#111110" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${mono.variable}`}>
      <head>
        {/* Warm DNS + TLS to the analytics origin so the beacon costs nothing later. */}
        <link rel="preconnect" href="https://va.vercel-scripts.com" crossOrigin="" />
      </head>
      <body>
        {children}
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
