"use client"

import React, { useState } from "react"
import { cn } from "@/lib/utils"

interface LeverageSliderProps {
  value: number
  min?: number
  max?: number
  step?: number
  onChange?: (value: number) => void
  className?: string
}

export function LeverageSlider({
  value,
  min = 1,
  max = 100,
  step = 1,
  onChange,
  className
}: LeverageSliderProps) {
  const [isHovering, setIsHovering] = useState(false)
  
  // Calculate risk based on leverage
  const getRiskLevel = (leverage: number) => {
    if (leverage <= 2) return "low"
    if (leverage <= 10) return "medium"
    if (leverage <= 25) return "high"
    return "extreme"
  }
  
  const riskLevel = getRiskLevel(value)
  
  const getRiskColor = () => {
    switch (riskLevel) {
      case "low":
        return "bg-green-500"
      case "medium":
        return "bg-yellow-500"
      case "high":
        return "bg-orange-500"
      case "extreme":
        return "bg-red-500"
      default:
        return "bg-gray-500"
    }
  }
  
  // Calculate percentage filled
  const percentage = ((value - min) / (max - min)) * 100
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value, 10)
    onChange?.(newValue)
  }
  
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">Leverage</label>
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold">{value}x</span>
          <span className={cn(
            "text-xs px-1.5 py-0.5 rounded-full",
            {
              "bg-green-100 text-green-800": riskLevel === "low",
              "bg-yellow-100 text-yellow-800": riskLevel === "medium",
              "bg-orange-100 text-orange-800": riskLevel === "high",
              "bg-red-100 text-red-800": riskLevel === "extreme",
            }
          )}>
            {riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)} Risk
          </span>
        </div>
      </div>
      
      <div className="relative">
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className={cn("h-full transition-all", getRiskColor())} 
            style={{ width: `${percentage}%` }} 
          />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          onMouseEnter={() => setIsHovering(true)}
          onMouseLeave={() => setIsHovering(false)}
        />
        
        {/* Tick marks */}
        <div className="flex justify-between mt-1 text-xs text-gray-500">
          <span>{min}x</span>
          {max > 20 && <span>{Math.round(max * 0.25)}x</span>}
          {max > 10 && <span>{Math.round(max * 0.5)}x</span>}
          {max > 20 && <span>{Math.round(max * 0.75)}x</span>}
          <span>{max}x</span>
        </div>
      </div>
      
      {isHovering && (
        <div className="text-xs text-muted-foreground">
          {riskLevel === "low" && "Conservative approach with minimal risk of liquidation."}
          {riskLevel === "medium" && "Balanced approach with moderate risk of liquidation."}
          {riskLevel === "high" && "Aggressive approach with significant risk of liquidation."}
          {riskLevel === "extreme" && "Maximum risk approach with very high chance of liquidation."}
        </div>
      )}
    </div>
  )
} 