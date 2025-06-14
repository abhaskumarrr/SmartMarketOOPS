'use client';

import React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export type TimeframeOption = {
  value: string;
  label: string;
};

// Default timeframe options
export const DEFAULT_TIMEFRAMES: TimeframeOption[] = [
  { value: '1m', label: '1m' },
  { value: '5m', label: '5m' },
  { value: '15m', label: '15m' },
  { value: '30m', label: '30m' },
  { value: '1h', label: '1h' },
  { value: '4h', label: '4h' },
  { value: '1d', label: '1D' },
  { value: '1w', label: '1W' },
  { value: '1M', label: '1M' },
];

interface TimeframeSelectorProps {
  timeframes?: TimeframeOption[];
  selectedTimeframe: string;
  onTimeframeChange: (timeframe: string) => void;
  className?: string;
}

export function TimeframeSelector({
  timeframes = DEFAULT_TIMEFRAMES,
  selectedTimeframe,
  onTimeframeChange,
  className
}: TimeframeSelectorProps) {
  return (
    <div className={cn("flex space-x-1", className)}>
      {timeframes.map((timeframe) => (
        <Button
          key={timeframe.value}
          variant={selectedTimeframe === timeframe.value ? "default" : "outline"}
          size="sm"
          className={cn(
            "h-7 px-2.5 text-xs font-medium", 
            selectedTimeframe === timeframe.value ? "" : "text-muted-foreground"
          )}
          onClick={() => onTimeframeChange(timeframe.value)}
        >
          {timeframe.label}
        </Button>
      ))}
    </div>
  );
} 