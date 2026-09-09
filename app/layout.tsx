import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LunarReg · SIH 2026",
  description:
    "Sub-pixel lunar image registration for Chandrayaan-2 and reference imagery.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
