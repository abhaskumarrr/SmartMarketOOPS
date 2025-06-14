"use client"

import React, { useEffect, useState, useRef } from "react"
import { cn } from "@/lib/utils"

interface PriceDisplayProps {
  value: number
  previousValue?: number
  decimals?: number
  prefix?: string
  suffix?: string
  className?: string
  flashOnChange?: boolean
  size?: "sm" | "md" | "lg"
}

export function PriceDisplay({
  value,
  previousValue,
  decimals = 2,
  prefix = "",
  suffix = "",
  className,
  flashOnChange = true,
  size = "md"
}: PriceDisplayProps) {
  const [flashClass, setFlashClass] = useState<string>("")
  const prevValueRef = useRef<number | undefined>(previousValue)
  
  // Format the price with the specified number of decimals
  const formattedPrice = value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  })

  // Determine size class
  const sizeClass = {
    sm: "text-sm",
    md: "text-base",
    lg: "text-lg font-semibold"
  }[size]

  useEffect(() => {
    // Only flash if flashOnChange is enabled and we have a previous value
    if (flashOnChange && prevValueRef.current !== undefined && value !== prevValueRef.current) {
      // Determine if the price went up or down
      const direction = value > prevValueRef.current ? "up" : "down"
      setFlashClass(direction === "up" ? "animate-price-flash-up text-profit" : "animate-price-flash-down text-loss")
      
      // Clear the flash after animation completes
      const timer = setTimeout(() => {
        setFlashClass("")
      }, 600)
      
      return () => clearTimeout(timer)
    }
    
    // Update the previous value ref
    prevValueRef.current = value
  }, [value, flashOnChange])

  return (
    <span 
      className={cn(
        "transition-colors rounded px-1 -mx-1",
        flashClass,
        sizeClass,
        className
      )}
    >
      {prefix}{formattedPrice}{suffix}
    </span>
  )
} 