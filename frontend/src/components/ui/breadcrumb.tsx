"use client"

import * as React from "react"
import Link from "next/link"
import { ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"

export interface BreadcrumbSegment {
  title: string
  href: string
}

interface BreadcrumbProps {
  segments: BreadcrumbSegment[]
  className?: string
}

export function Breadcrumb({ segments, className }: BreadcrumbProps) {
  if (!segments || segments.length === 0) {
    return null
  }

  return (
    <nav aria-label="Breadcrumb" className={cn("flex", className)}>
      <ol className="flex items-center text-sm text-muted-foreground">
        <li>
          <Link 
            href="/" 
            className="transition-colors hover:text-foreground"
          >
            Home
          </Link>
        </li>
        
        {segments.map((segment, index) => (
          <li key={index} className="flex items-center">
            <ChevronRight className="mx-1 h-4 w-4" />
            {index === segments.length - 1 ? (
              <span className="font-medium text-foreground">
                {segment.title}
              </span>
            ) : (
              <Link 
                href={segment.href} 
                className="transition-colors hover:text-foreground"
              >
                {segment.title}
              </Link>
            )}
          </li>
        ))}
      </ol>
    </nav>
  )
} 