'use client';

import React, { useState } from 'react';
import { 
  BarChart2, 
  LineChart, 
  CandlestickChart, 
  Grid, 
  Settings,
  Download,
  Fullscreen,
  Maximize2,
  Crosshair,
  Ruler
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { TimeframeSelector } from './TimeframeSelector';
import { 
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider 
} from '@/components/ui/tooltip';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuCheckboxItem,
} from '@/components/ui/dropdown-menu';
import { Toggle } from '@/components/ui/toggle';
import { Separator } from '@/components/ui/separator';

export type ChartType = 'candle' | 'line' | 'bar';

interface ChartToolbarProps {
  chartType: ChartType;
  onChartTypeChange: (type: ChartType) => void;
  timeframe: string;
  onTimeframeChange: (timeframe: string) => void;
  showVolume: boolean;
  onVolumeToggle: (show: boolean) => void;
  showGrid: boolean;
  onGridToggle: (show: boolean) => void;
  onFullscreen: () => void;
  onExportImage: () => void;
  onOpenSettings: () => void;
  className?: string;
  autoHideToolbar?: boolean;
}

export function ChartToolbar({
  chartType,
  onChartTypeChange,
  timeframe,
  onTimeframeChange,
  showVolume,
  onVolumeToggle,
  showGrid,
  onGridToggle,
  onFullscreen,
  onExportImage,
  onOpenSettings,
  className,
  autoHideToolbar = false
}: ChartToolbarProps) {
  const [isHovered, setIsHovered] = useState(!autoHideToolbar);

  return (
    <div 
      className={`relative transition-opacity duration-200 ${className} ${
        autoHideToolbar && !isHovered ? 'opacity-0' : 'opacity-100'
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => autoHideToolbar && setIsHovered(false)}
    >
      <div className="absolute top-2 left-2 right-2 z-10 rounded-md bg-background/80 backdrop-blur-sm border shadow-sm flex items-center justify-between p-1">
        <div className="flex items-center space-x-1">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Toggle 
                  size="sm"
                  pressed={chartType === 'candle'} 
                  onPressedChange={() => onChartTypeChange('candle')}
                  className="h-7 w-7 p-0"
                >
                  <CandlestickChart className="h-3.5 w-3.5" />
                </Toggle>
              </TooltipTrigger>
              <TooltipContent side="bottom">Candlestick Chart</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Toggle 
                  size="sm"
                  pressed={chartType === 'line'} 
                  onPressedChange={() => onChartTypeChange('line')}
                  className="h-7 w-7 p-0"
                >
                  <LineChart className="h-3.5 w-3.5" />
                </Toggle>
              </TooltipTrigger>
              <TooltipContent side="bottom">Line Chart</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Toggle 
                  size="sm"
                  pressed={chartType === 'bar'} 
                  onPressedChange={() => onChartTypeChange('bar')}
                  className="h-7 w-7 p-0"
                >
                  <BarChart2 className="h-3.5 w-3.5" />
                </Toggle>
              </TooltipTrigger>
              <TooltipContent side="bottom">Bar Chart</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <Separator orientation="vertical" className="mx-1 h-6" />
          
          <TimeframeSelector
            selectedTimeframe={timeframe}
            onTimeframeChange={onTimeframeChange}
          />
        </div>
        
        <div className="flex items-center space-x-1">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Toggle 
                  size="sm"
                  pressed={showVolume} 
                  onPressedChange={onVolumeToggle}
                  className="h-7 w-7 p-0"
                >
                  <BarChart2 className="h-3.5 w-3.5" />
                </Toggle>
              </TooltipTrigger>
              <TooltipContent side="bottom">Toggle Volume</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Toggle 
                  size="sm"
                  pressed={showGrid} 
                  onPressedChange={onGridToggle}
                  className="h-7 w-7 p-0"
                >
                  <Grid className="h-3.5 w-3.5" />
                </Toggle>
              </TooltipTrigger>
              <TooltipContent side="bottom">Toggle Grid</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="icon" variant="ghost" className="h-7 w-7">
                <Crosshair className="h-3.5 w-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Drawing Tools</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <Ruler className="mr-2 h-4 w-4" />
                <span>Trend Line</span>
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Maximize2 className="mr-2 h-4 w-4" />
                <span>Fibonacci Retracement</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <span>Clear All Drawings</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={onExportImage}>
                  <Download className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">Export Chart as Image</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={onFullscreen}>
                  <Fullscreen className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">Fullscreen</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={onOpenSettings}>
                  <Settings className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">Chart Settings</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
    </div>
  );
} 