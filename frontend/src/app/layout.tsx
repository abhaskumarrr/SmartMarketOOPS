import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { VersionSkewProtection } from "@/components/VersionSkewProtection";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "SmartMarketOOPS - Professional Trading Dashboard",
  description: "AI-powered trading platform with real-time market analysis and automated trading capabilities",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased`}>
        <VersionSkewProtection>
          <div className="min-h-screen bg-background">
            <header className="border-b">
              <div className="flex h-14 items-center px-4 lg:px-6">
                <div className="flex-1">
                  <h1 className="text-lg font-semibold">SmartMarketOOPS</h1>
                </div>
              </div>
            </header>
            <main className="flex-1 p-4 lg:p-6">
              {children}
            </main>
          </div>
        </VersionSkewProtection>
      </body>
    </html>
  );
}
