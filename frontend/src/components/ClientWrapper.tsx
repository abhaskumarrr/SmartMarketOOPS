"use client"

import React from 'react'

interface ClientWrapperProps {
  children: React.ReactNode
}

export function ClientWrapper({ children }: ClientWrapperProps) {
  return (
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
  )
}
