"use client"

import React from 'react'
import { ErrorBoundary, AsyncErrorBoundary } from '@/components/ErrorBoundary'
import { SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import { AppSidebar } from '@/components/app-sidebar'

interface ClientWrapperProps {
  children: React.ReactNode
}

export function ClientWrapper({ children }: ClientWrapperProps) {
  return (
    <ErrorBoundary>
      <AsyncErrorBoundary>
        <SidebarProvider>
          <AppSidebar />
          <main className="flex-1">
            <div className="flex h-14 items-center border-b px-4 lg:px-6">
              <SidebarTrigger className="mr-4" />
              <div className="flex-1">
                <h1 className="text-lg font-semibold">SmartMarketOOPS</h1>
              </div>
            </div>
            <div className="flex-1 p-4 lg:p-6">
              <ErrorBoundary>
                {children}
              </ErrorBoundary>
            </div>
          </main>
        </SidebarProvider>
      </AsyncErrorBoundary>
    </ErrorBoundary>
  )
}
