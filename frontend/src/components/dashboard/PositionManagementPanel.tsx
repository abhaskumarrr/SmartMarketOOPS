'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../ui/table';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Input } from '../ui/input';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '../ui/dialog';
import { Label } from '../ui/label';
import { 
  TrendingUp, 
  TrendingDown, 
  ArrowRight, 
  X, 
  Edit,
  Shield,
  DollarSign,
  AlertTriangle,
  Loader2
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useRealTimeData } from '@/hooks/useRealTimeData';
import { toast } from '../ui/use-toast';
import axios from 'axios';

interface PositionManagementPanelProps {
  className?: string;
  compact?: boolean; // Optional compact mode for smaller screens
}

interface Position {
  id: number;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  liquidationPrice: number | null;
  leverage: number;
  pnl: number;
  pnlPercentage: number;
  stopLoss: number | null;
  takeProfit: number | null;
}

interface ModifyPositionParams {
  positionId: number;
  stopLoss?: number | null;
  takeProfit?: number | null;
  action?: 'close' | 'modify';
}

export default function PositionManagementPanel({ className, compact = false }: PositionManagementPanelProps) {
  const { isConnected, lastMarketData } = useRealTimeData();
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null);
  const [modifyDialogOpen, setModifyDialogOpen] = useState(false);
  const [isModifying, setIsModifying] = useState(false);
  const [modifyParams, setModifyParams] = useState<{
    stopLoss: string;
    takeProfit: string;
  }>({
    stopLoss: '',
    takeProfit: '',
  });

  // Fetch positions on component mount and when market data updates
  useEffect(() => {
    const fetchPositions = async () => {
      try {
        setLoading(true);
        const response = await axios.get('/api/positions');
        
        if (response.data && response.data.success) {
          setPositions(response.data.data);
        } else {
          console.error('Failed to fetch positions:', response.data?.message || 'Unknown error');
          // If API fails, use mock data during development
          setPositions(getMockPositions());
        }
      } catch (error) {
        console.error('Error fetching positions:', error);
        // If API fails, use mock data during development
        setPositions(getMockPositions());
      } finally {
        setLoading(false);
      }
    };

    fetchPositions();

    // Set up polling for position updates
    const interval = setInterval(fetchPositions, 30000); // Every 30 seconds

    return () => clearInterval(interval);
  }, []);

  // Update position prices when market data changes
  useEffect(() => {
    if (!lastMarketData || Object.keys(lastMarketData).length === 0) {
      return;
    }

    setPositions(prevPositions => 
      prevPositions.map(position => {
        const marketData = lastMarketData[position.symbol];
        if (!marketData) return position;

        const currentPrice = marketData.price;
        const priceDiff = position.side === 'long' 
          ? currentPrice - position.entryPrice 
          : position.entryPrice - currentPrice;
        const pnl = priceDiff * position.size;
        const pnlPercentage = (priceDiff / position.entryPrice) * 100 * position.leverage;

        return {
          ...position,
          currentPrice,
          pnl,
          pnlPercentage
        };
      })
    );
  }, [lastMarketData]);

  // Generate mock positions for development
  const getMockPositions = (): Position[] => {
    return [
      {
        id: 1,
        symbol: 'BTCUSD',
        side: 'long',
        size: 0.25,
        entryPrice: 47650.75,
        currentPrice: 48250.45,
        liquidationPrice: 42150.35,
        leverage: 10,
        pnl: 149.93,
        pnlPercentage: 12.55,
        stopLoss: 46500.0,
        takeProfit: 52000.0
      },
      {
        id: 2,
        symbol: 'ETHUSD',
        side: 'short',
        size: 1.5,
        entryPrice: 2950.25,
        currentPrice: 2870.12,
        liquidationPrice: 3245.60,
        leverage: 5,
        pnl: 120.20,
        pnlPercentage: 4.08,
        stopLoss: 3050.0,
        takeProfit: 2600.0
      },
      {
        id: 3,
        symbol: 'SOLUSD',
        side: 'long',
        size: 10,
        entryPrice: 103.45,
        currentPrice: 106.78,
        liquidationPrice: 91.23,
        leverage: 8,
        pnl: 33.30,
        pnlPercentage: 25.66,
        stopLoss: null,
        takeProfit: null
      }
    ];
  };

  // Format currency with sign
  const formatCurrency = (value: number, digits: number = 2): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      signDisplay: 'always'
    }).format(value);
  };

  // Handle position modification
  const handleModifyPosition = (position: Position) => {
    setSelectedPosition(position);
    setModifyParams({
      stopLoss: position.stopLoss?.toString() || '',
      takeProfit: position.takeProfit?.toString() || '',
    });
    setModifyDialogOpen(true);
  };

  // Handle position close
  const handleClosePosition = async (positionId: number) => {
    try {
      setIsModifying(true);
      
      const response = await axios.post('/api/positions/modify', {
        positionId,
        action: 'close'
      });
      
      if (response.data && response.data.success) {
        toast({
          title: 'Position closed',
          description: 'Position has been closed successfully.',
          variant: 'default',
        });
        
        // Remove position from list
        setPositions(prev => prev.filter(p => p.id !== positionId));
      } else {
        throw new Error(response.data?.message || 'Failed to close position');
      }
    } catch (error) {
      console.error('Error closing position:', error);
      
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to close position',
        variant: 'destructive',
      });
    } finally {
      setIsModifying(false);
    }
  };

  // Handle save position modifications
  const handleSaveModifications = async () => {
    if (!selectedPosition) return;
    
    try {
      setIsModifying(true);
      
      const stopLoss = modifyParams.stopLoss === '' ? null : parseFloat(modifyParams.stopLoss);
      const takeProfit = modifyParams.takeProfit === '' ? null : parseFloat(modifyParams.takeProfit);
      
      const response = await axios.post('/api/positions/modify', {
        positionId: selectedPosition.id,
        stopLoss,
        takeProfit,
        action: 'modify'
      });
      
      if (response.data && response.data.success) {
        toast({
          title: 'Position updated',
          description: 'Stop loss and take profit have been updated.',
          variant: 'default',
        });
        
        // Update position in list
        setPositions(prev => 
          prev.map(p => 
            p.id === selectedPosition.id 
              ? { ...p, stopLoss, takeProfit } 
              : p
          )
        );
        
        setModifyDialogOpen(false);
      } else {
        throw new Error(response.data?.message || 'Failed to update position');
      }
    } catch (error) {
      console.error('Error updating position:', error);
      
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to update position',
        variant: 'destructive',
      });
    } finally {
      setIsModifying(false);
    }
  };

  return (
    <div className={cn("w-full", className)}>
      {loading ? (
        <div className={cn(
          "flex items-center justify-center",
          compact ? "h-40" : "h-64"
        )}>
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
          <span className="ml-2">Loading positions...</span>
        </div>
      ) : positions.length === 0 ? (
        <div className={cn(
          "flex flex-col items-center justify-center text-center text-muted-foreground",
          compact ? "p-4 h-40" : "p-6 h-64"
        )}>
          <Shield className={cn("mb-2", compact ? "h-8 w-8" : "h-12 w-12")} />
          <p className={compact ? "text-sm" : "text-base"}>No open positions</p>
          <p className={compact ? "text-xs" : "text-sm"}>Your active trades will appear here</p>
        </div>
      ) : (
        <div className={compact ? "overflow-x-auto" : ""}>
          <Table className={compact ? "text-sm" : ""}>
            <TableHeader>
              <TableRow>
                <TableHead className={compact ? "py-1 px-2 text-xs" : ""}>Symbol</TableHead>
                <TableHead className={cn("text-right", compact ? "py-1 px-2 text-xs" : "")}>Size</TableHead>
                <TableHead className={cn("text-right", compact ? "py-1 px-2 text-xs hidden" : "")}>Entry</TableHead>
                <TableHead className={cn("text-right", compact ? "py-1 px-2 text-xs" : "")}>PnL</TableHead>
                <TableHead className={cn("text-right", compact ? "py-1 px-2 text-xs" : "")}>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positions.map((position) => (
                <TableRow key={position.id}>
                  <TableCell className={compact ? "py-1 px-2" : ""}>
                    <div className="flex flex-col">
                      <div className="flex items-center">
                        <Badge variant={position.side === 'long' ? 'default' : 'destructive'} className="mr-1">
                          {position.side === 'long' ? 'L' : 'S'}
                        </Badge>
                        <span className={compact ? "text-xs font-medium" : "text-sm font-medium"}>
                          {position.symbol}
                        </span>
                      </div>
                      {!compact && (
                        <div className="text-xs text-muted-foreground mt-1">
                          Lev: {position.leverage}x
                        </div>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className={cn("text-right", compact ? "py-1 px-2" : "")}>
                    <div className="flex flex-col items-end">
                      <span className={compact ? "text-xs font-medium" : "text-sm font-medium"}>
                        {position.size}
                      </span>
                      {!compact && (
                        <span className="text-xs text-muted-foreground">
                          ${(position.size * position.currentPrice).toFixed(2)}
                        </span>
                      )}
                    </div>
                  </TableCell>
                  {!compact && (
                    <TableCell className="text-right">
                      <div className="flex flex-col items-end">
                        <span className="text-sm font-medium">
                          ${position.entryPrice.toFixed(2)}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          Curr: ${position.currentPrice.toFixed(2)}
                        </span>
                      </div>
                    </TableCell>
                  )}
                  <TableCell className={cn("text-right", compact ? "py-1 px-2" : "")}>
                    <div className="flex flex-col items-end">
                      <span className={cn(
                        compact ? "text-xs font-medium" : "text-sm font-medium",
                        position.pnl >= 0 ? "text-green-500" : "text-red-500"
                      )}>
                        {position.pnl >= 0 ? '+' : ''}{position.pnl.toFixed(2)}
                      </span>
                      <span className={cn(
                        "text-xs",
                        position.pnlPercentage >= 0 ? "text-green-500" : "text-red-500"
                      )}>
                        {position.pnlPercentage >= 0 ? '+' : ''}{position.pnlPercentage.toFixed(2)}%
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className={cn("text-right", compact ? "py-1 px-2" : "")}>
                    <div className="flex justify-end gap-1">
                      <Button 
                        variant="outline" 
                        size={compact ? "icon" : "sm"}
                        className={compact ? "h-7 w-7" : ""}
                        onClick={() => handleModifyPosition(position)}
                      >
                        <Edit className={compact ? "h-3 w-3" : "h-4 w-4"} />
                        {!compact && <span className="ml-1">Edit</span>}
                      </Button>
                      <Button 
                        variant="destructive" 
                        size={compact ? "icon" : "sm"}
                        className={compact ? "h-7 w-7" : ""}
                        onClick={() => handleClosePosition(position.id)}
                        disabled={isModifying}
                      >
                        <X className={compact ? "h-3 w-3" : "h-4 w-4"} />
                        {!compact && <span className="ml-1">Close</span>}
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {/* Modify Position Dialog */}
      <Dialog open={modifyDialogOpen} onOpenChange={setModifyDialogOpen}>
        <DialogContent className={compact ? "max-w-[90vw] sm:max-w-md" : "max-w-md"}>
          <DialogHeader>
            <DialogTitle>Modify Position</DialogTitle>
            <DialogDescription>
              {selectedPosition && (
                <div className="mt-2">
                  <div className="flex justify-between items-center">
                    <Badge variant={selectedPosition.side === 'long' ? 'default' : 'destructive'}>
                      {selectedPosition.side.toUpperCase()}
                    </Badge>
                    <span className="font-medium">{selectedPosition.symbol}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Size</p>
                      <p className="text-base font-medium">{selectedPosition.size}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Entry Price</p>
                      <p className="text-base font-medium">${selectedPosition.entryPrice.toFixed(2)}</p>
                    </div>
                  </div>
                </div>
              )}
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="stopLoss">Stop Loss (USD)</Label>
              <Input
                id="stopLoss"
                type="number"
                value={modifyParams.stopLoss}
                onChange={(e) => setModifyParams({...modifyParams, stopLoss: e.target.value})}
                placeholder="Optional stop loss price"
                step="0.01"
                min="0"
                disabled={isModifying}
              />
              {selectedPosition?.side === 'long' && parseFloat(modifyParams.stopLoss) > 0 && (
                <p className="text-xs text-muted-foreground">
                  {parseFloat(modifyParams.stopLoss) >= selectedPosition.entryPrice ? 
                    <span className="text-amber-500">Warning: Stop loss is above entry price</span> :
                    `${((1 - parseFloat(modifyParams.stopLoss) / selectedPosition.entryPrice) * 100).toFixed(2)}% loss from entry`
                  }
                </p>
              )}
              {selectedPosition?.side === 'short' && parseFloat(modifyParams.stopLoss) > 0 && (
                <p className="text-xs text-muted-foreground">
                  {parseFloat(modifyParams.stopLoss) <= selectedPosition.entryPrice ? 
                    <span className="text-amber-500">Warning: Stop loss is below entry price</span> :
                    `${((parseFloat(modifyParams.stopLoss) / selectedPosition.entryPrice - 1) * 100).toFixed(2)}% loss from entry`
                  }
                </p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="takeProfit">Take Profit (USD)</Label>
              <Input
                id="takeProfit"
                type="number"
                value={modifyParams.takeProfit}
                onChange={(e) => setModifyParams({...modifyParams, takeProfit: e.target.value})}
                placeholder="Optional take profit price"
                step="0.01"
                min="0"
                disabled={isModifying}
              />
              {selectedPosition?.side === 'long' && parseFloat(modifyParams.takeProfit) > 0 && (
                <p className="text-xs text-muted-foreground">
                  {parseFloat(modifyParams.takeProfit) <= selectedPosition.entryPrice ? 
                    <span className="text-amber-500">Warning: Take profit is below entry price</span> :
                    `${((parseFloat(modifyParams.takeProfit) / selectedPosition.entryPrice - 1) * 100).toFixed(2)}% profit from entry`
                  }
                </p>
              )}
              {selectedPosition?.side === 'short' && parseFloat(modifyParams.takeProfit) > 0 && (
                <p className="text-xs text-muted-foreground">
                  {parseFloat(modifyParams.takeProfit) >= selectedPosition.entryPrice ? 
                    <span className="text-amber-500">Warning: Take profit is above entry price</span> :
                    `${((1 - parseFloat(modifyParams.takeProfit) / selectedPosition.entryPrice) * 100).toFixed(2)}% profit from entry`
                  }
                </p>
              )}
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setModifyDialogOpen(false)} disabled={isModifying}>
              Cancel
            </Button>
            <Button onClick={handleSaveModifications} disabled={isModifying}>
              {isModifying ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  Save Changes
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
} 