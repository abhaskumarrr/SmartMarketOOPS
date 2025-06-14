'use client';

import React, { useState } from 'react';
import { 
  Popover, 
  PopoverContent, 
  PopoverTrigger 
} from '@/components/ui/popover';
import { Button } from '@/components/ui/button';
import { 
  Command, 
  CommandEmpty, 
  CommandGroup, 
  CommandInput, 
  CommandItem, 
  CommandList, 
  CommandSeparator 
} from '@/components/ui/command';
import { Check, ChevronDown, Plus, Settings, X } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger, 
  DialogFooter 
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { ChartIndicator } from '@/types';

// Available technical indicators
const availableIndicators = [
  { id: 'sma', name: 'Simple Moving Average', type: 'overlay', category: 'trend', defaultParams: { period: 20 } },
  { id: 'ema', name: 'Exponential Moving Average', type: 'overlay', category: 'trend', defaultParams: { period: 9 } },
  { id: 'bollinger', name: 'Bollinger Bands', type: 'overlay', category: 'volatility', defaultParams: { period: 20, stdDev: 2 } },
  { id: 'macd', name: 'MACD', type: 'oscillator', category: 'momentum', defaultParams: { fast: 12, slow: 26, signal: 9 } },
  { id: 'rsi', name: 'Relative Strength Index', type: 'oscillator', category: 'momentum', defaultParams: { period: 14 } },
  { id: 'stoch', name: 'Stochastic Oscillator', type: 'oscillator', category: 'momentum', defaultParams: { k: 14, d: 3, smooth: 3 } },
  { id: 'obv', name: 'On-Balance Volume', type: 'oscillator', category: 'volume', defaultParams: {} },
  { id: 'adx', name: 'Average Directional Index', type: 'oscillator', category: 'trend', defaultParams: { period: 14 } },
  { id: 'atr', name: 'Average True Range', type: 'oscillator', category: 'volatility', defaultParams: { period: 14 } },
  { id: 'cci', name: 'Commodity Channel Index', type: 'oscillator', category: 'momentum', defaultParams: { period: 20 } },
  { id: 'ichimoku', name: 'Ichimoku Cloud', type: 'overlay', category: 'trend', defaultParams: { conversionPeriod: 9, basePeriod: 26, spanPeriod: 52, displacement: 26 } },
  { id: 'vwap', name: 'Volume Weighted Average Price', type: 'overlay', category: 'volume', defaultParams: { period: 14 } },
];

interface IndicatorSelectorProps {
  activeIndicators: ChartIndicator[];
  onAddIndicator: (indicator: ChartIndicator) => void;
  onRemoveIndicator: (indicatorId: string) => void;
  onUpdateIndicator: (indicatorId: string, params: Record<string, number>) => void;
}

export function IndicatorSelector({
  activeIndicators,
  onAddIndicator,
  onRemoveIndicator,
  onUpdateIndicator
}: IndicatorSelectorProps) {
  const [open, setOpen] = useState(false);
  const [configOpen, setConfigOpen] = useState(false);
  const [selectedIndicator, setSelectedIndicator] = useState<ChartIndicator | null>(null);
  const [tempParams, setTempParams] = useState<Record<string, number>>({});
  
  // Filter out already added indicators
  const filteredIndicators = availableIndicators.filter(
    indicator => !activeIndicators.some(active => active.id === indicator.id)
  );

  const handleAddIndicator = (indicator: any) => {
    const newIndicator: ChartIndicator = {
      id: indicator.id,
      name: indicator.name,
      type: indicator.type,
      params: { ...indicator.defaultParams },
      color: getRandomColor(),
      enabled: true
    };
    
    onAddIndicator(newIndicator);
    setOpen(false);
  };

  const handleOpenConfig = (indicator: ChartIndicator) => {
    setSelectedIndicator(indicator);
    setTempParams({ ...indicator.params });
    setConfigOpen(true);
  };

  const handleSaveConfig = () => {
    if (selectedIndicator) {
      onUpdateIndicator(selectedIndicator.id, tempParams);
      setConfigOpen(false);
    }
  };

  // Generate a random color for the indicator
  const getRandomColor = () => {
    const colors = [
      '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
      '#FF9F40', '#2ECC71', '#F1C40F', '#E74C3C', '#3498DB',
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Indicators</h3>
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="h-8 gap-1">
              <Plus className="h-3.5 w-3.5" />
              Add
              <ChevronDown className="h-3.5 w-3.5 opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="p-0" align="end" side="bottom" sideOffset={8} alignOffset={0} width={320}>
            <Command>
              <CommandInput placeholder="Search indicators..." />
              <CommandList>
                <CommandEmpty>No indicators found.</CommandEmpty>
                <CommandGroup heading="Trend">
                  {filteredIndicators
                    .filter(indicator => indicator.category === 'trend')
                    .map(indicator => (
                      <CommandItem 
                        key={indicator.id}
                        onSelect={() => handleAddIndicator(indicator)}
                        className="flex items-center gap-2"
                      >
                        <span>{indicator.name}</span>
                        <span className="ml-auto text-xs text-muted-foreground">{indicator.type}</span>
                      </CommandItem>
                    ))
                  }
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="Momentum">
                  {filteredIndicators
                    .filter(indicator => indicator.category === 'momentum')
                    .map(indicator => (
                      <CommandItem 
                        key={indicator.id}
                        onSelect={() => handleAddIndicator(indicator)}
                        className="flex items-center gap-2"
                      >
                        <span>{indicator.name}</span>
                        <span className="ml-auto text-xs text-muted-foreground">{indicator.type}</span>
                      </CommandItem>
                    ))
                  }
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="Volatility">
                  {filteredIndicators
                    .filter(indicator => indicator.category === 'volatility')
                    .map(indicator => (
                      <CommandItem 
                        key={indicator.id}
                        onSelect={() => handleAddIndicator(indicator)}
                        className="flex items-center gap-2"
                      >
                        <span>{indicator.name}</span>
                        <span className="ml-auto text-xs text-muted-foreground">{indicator.type}</span>
                      </CommandItem>
                    ))
                  }
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="Volume">
                  {filteredIndicators
                    .filter(indicator => indicator.category === 'volume')
                    .map(indicator => (
                      <CommandItem 
                        key={indicator.id}
                        onSelect={() => handleAddIndicator(indicator)}
                        className="flex items-center gap-2"
                      >
                        <span>{indicator.name}</span>
                        <span className="ml-auto text-xs text-muted-foreground">{indicator.type}</span>
                      </CommandItem>
                    ))
                  }
                </CommandGroup>
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      </div>

      <ScrollArea className="h-[120px] rounded-md border p-2">
        {activeIndicators.length > 0 ? (
          <div className="space-y-2">
            {activeIndicators.map((indicator) => (
              <div 
                key={indicator.id} 
                className="flex items-center justify-between rounded-md border p-2"
                style={{ borderLeftColor: indicator.color, borderLeftWidth: 3 }}
              >
                <div className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: indicator.color }}
                  />
                  <span className="text-sm font-medium">{indicator.name}</span>
                  <Badge variant="outline" className="text-xs">
                    {indicator.type === 'overlay' ? 'Overlay' : 'Indicator'}
                  </Badge>
                </div>
                <div className="flex items-center gap-1">
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="h-7 w-7" 
                    onClick={() => handleOpenConfig(indicator)}
                  >
                    <Settings className="h-3.5 w-3.5" />
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="h-7 w-7" 
                    onClick={() => onRemoveIndicator(indicator.id)}
                  >
                    <X className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
            No indicators added. Click "Add" to add technical indicators.
          </div>
        )}
      </ScrollArea>

      {/* Indicator Configuration Dialog */}
      <Dialog open={configOpen} onOpenChange={setConfigOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Configure {selectedIndicator?.name}</DialogTitle>
          </DialogHeader>
          {selectedIndicator && (
            <div className="grid gap-4 py-4">
              {Object.entries(selectedIndicator.params).map(([key, value]) => (
                <div key={key} className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor={key} className="text-right capitalize">
                    {key}
                  </Label>
                  <Input
                    id={key}
                    type="number"
                    className="col-span-3"
                    value={tempParams[key] || value}
                    onChange={(e) => 
                      setTempParams({
                        ...tempParams,
                        [key]: parseFloat(e.target.value)
                      })
                    }
                  />
                </div>
              ))}
            </div>
          )}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setConfigOpen(false)}>
              Cancel
            </Button>
            <Button type="button" onClick={handleSaveConfig}>
              Save changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
} 