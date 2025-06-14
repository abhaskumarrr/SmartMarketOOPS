"use client"

import * as React from "react"
import { usePathname } from "next/navigation"
import { Breadcrumb } from "@/components/ui/breadcrumb"
import { getBreadcrumbSegments, getPageTitle } from "@/config/navigation"

export function PageHeader() {
  const pathname = usePathname()
  const segments = getBreadcrumbSegments(pathname)
  const pageTitle = getPageTitle(pathname)

  // Transform segments to match Breadcrumb component props
  const breadcrumbSegments = segments.map(segment => ({
    title: segment.name,
    href: segment.href
  }))

  return (
    <div className="border-b bg-background px-6 py-3">
      <Breadcrumb segments={breadcrumbSegments} />
      <h1 className="mt-2 text-2xl font-bold tracking-tight">{pageTitle}</h1>
    </div>
  )
} 