import { ReactNode } from 'react';

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="min-h-screen bg-background">
      <div className="flex">
        {/* Sidebar will go here */}
        <aside className="w-64 bg-card border-r border-border">
          <div className="p-6">
            <h2 className="text-lg font-semibold text-foreground">SmartMarketOOPS</h2>
            <p className="text-sm text-muted-foreground">Trading Dashboard</p>
          </div>
          
          <nav className="px-4 space-y-2">
            <a href="/" className="block px-4 py-2 text-sm text-foreground hover:bg-accent rounded-md">
              Home
            </a>
            <a href="/dashboard" className="block px-4 py-2 text-sm text-foreground hover:bg-accent rounded-md">
              Overview
            </a>
            <a href="/dashboard/positions" className="block px-4 py-2 text-sm text-foreground hover:bg-accent rounded-md">
              Positions
            </a>
            <a href="/charts" className="block px-4 py-2 text-sm text-foreground hover:bg-accent rounded-md">
              Live Charts
            </a>
            <a href="/analytics" className="block px-4 py-2 text-sm text-foreground hover:bg-accent rounded-md">
              AI Analytics
            </a>
          </nav>
        </aside>

        {/* Main content */}
        <main className="flex-1">
          <header className="bg-card border-b border-border px-6 py-4">
            <div className="flex items-center justify-between">
              <h1 className="text-2xl font-bold text-foreground">Trading Dashboard</h1>
              <div className="flex items-center space-x-4">
                <div className="text-sm text-muted-foreground">
                  Last updated: {new Date().toLocaleTimeString()}
                </div>
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" title="Live data" />
              </div>
            </div>
          </header>
          
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
