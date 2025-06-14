"use client"

import React from "react"
import { Button, ButtonProps } from "@/components/ui/button"
import { cn } from "@/lib/utils"

type TradeType = "buy" | "sell" | "neutral"

interface TradeButtonProps extends Omit<ButtonProps, "variant"> {
  tradeType: TradeType
  loading?: boolean
  pulse?: boolean
  size?: "default" | "sm" | "lg" | "icon"
}

export function TradeButton({
  tradeType,
  loading = false,
  pulse = false,
  size = "default",
  className,
  children,
  ...props
}: TradeButtonProps) {
  // Get the appropriate classes based on the trade type
  const getTradeClasses = (type: TradeType) => {
    switch (type) {
      case "buy":
        return {
          base: "bg-profit hover:bg-profit-light text-white",
          pulse: pulse ? "animate-pulse-buy" : "",
        }
      case "sell":
        return {
          base: "bg-loss hover:bg-loss-light text-white",
          pulse: pulse ? "animate-pulse-sell" : "",
        }
      case "neutral":
      default:
        return {
          base: "",
          pulse: "",
        }
    }
  }

  const tradeClasses = getTradeClasses(tradeType)

  return (
    <Button
      variant={tradeType === "neutral" ? "default" : "none"}
      size={size}
      className={cn(
        tradeClasses.base,
        tradeClasses.pulse,
        className
      )}
      disabled={loading || props.disabled}
      {...props}
    >
      {loading ? (
        <span className="flex items-center gap-2">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          {children}
        </span>
      ) : (
        children
      )}
    </Button>
  )
} 