import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { VersionSkewProtection } from "@/components/VersionSkewProtection";
import { ThemeProvider } from "@/components/theme-provider";
import { ModeToggle } from "@/components/ui/mode-toggle";
import { Toaster } from "@/components/ui/toaster";
import { AppSidebar } from "@/components/app-sidebar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { MobileNav } from "@/components/ui/mobile-nav";
import { PageHeader } from "@/components/page-header";

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
      <body className={`${inter.className} ${inter.variable} antialiased`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <SidebarProvider>
            <div className="min-h-screen flex flex-col">
              {/* Header */}
              <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
                <div className="container flex h-14 items-center">
                  <div className="mr-4 hidden md:flex">
                    <ModeToggle />
                  </div>
                  <div className="flex flex-1 items-center justify-between">
                    <div className="flex items-center">
                      <MobileNav />
                      <div className="hidden md:block">
                        <h1 className="text-xl font-bold">SmartMarketOOPS</h1>
                      </div>
                    </div>
                    <nav className="flex items-center gap-2">
                      <div className="md:hidden">
                        <ModeToggle />
                      </div>
                      {/* Add profile or account items here */}
                    </nav>
                  </div>
                </div>
              </header>
              
              {/* Main content area with sidebar */}
              <div className="flex-1 flex">
                <AppSidebar className="hidden md:block" />
                <main className="flex-1 flex flex-col">
                  <PageHeader />
                  <div className="flex-1 p-6 pt-2">
                    {children}
                  </div>
                </main>
              </div>
            </div>
          </SidebarProvider>
          
          <Toaster />
          <VersionSkewProtection />
        </ThemeProvider>
      </body>
    </html>
  );
}
