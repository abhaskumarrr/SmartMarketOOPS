'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Settings, GripVertical, X, Plus, Save, Undo } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useBreakpoints } from '@/hooks/use-responsive';

// Widget types for the dashboard
export type WidgetType = 
  | 'chart' 
  | 'portfolio' 
  | 'positions' 
  | 'trade' 
  | 'watchlist'
  | 'news'
  | 'performance';

// Widget definition interface
export interface Widget {
  id: string;
  type: WidgetType;
  title: string;
  width: number; // 1-12 grid columns
  height: number; // Height in rows
  position: {
    x: number; // Grid position X
    y: number; // Grid position Y
  };
  settings?: Record<string, unknown>;
}

// Dashboard layout interface
export interface DashboardLayout {
  id: string;
  name: string;
  widgets: Widget[];
}

// Widget content mapping
const WidgetContent: Record<WidgetType, React.FC<{ widget: Widget, compact?: boolean }>> = {
  chart: ({ widget }) => (
    <div className="flex items-center justify-center h-full bg-muted/20 rounded-md">
      {widget.settings?.compact ? (
        <div className="text-sm text-center text-muted-foreground">Chart Widget (Compact)</div>
      ) : (
        <div className="text-center text-muted-foreground">Chart Widget</div>
      )}
    </div>
  ),
  portfolio: ({ widget }) => (
    <div className="flex items-center justify-center h-full bg-muted/20 rounded-md">
      {widget.settings?.compact ? (
        <div className="text-sm text-center text-muted-foreground">Portfolio Widget (Compact)</div>
      ) : (
        <div className="text-center text-muted-foreground">Portfolio Widget</div>
      )}
    </div>
  ),
  positions: ({ widget }) => (
    <div className="flex items-center justify-center h-full bg-muted/20 rounded-md">
      {widget.settings?.compact ? (
        <div className="text-sm text-center text-muted-foreground">Positions Widget (Compact)</div>
      ) : (
        <div className="text-center text-muted-foreground">Positions Widget</div>
      )}
    </div>
  ),
  trade: ({ widget }) => (
    <div className="flex items-center justify-center h-full bg-muted/20 rounded-md">
      {widget.settings?.compact ? (
        <div className="text-sm text-center text-muted-foreground">Trade Widget (Compact)</div>
      ) : (
        <div className="text-center text-muted-foreground">Trade Widget</div>
      )}
    </div>
  ),
  watchlist: ({ widget }) => (
    <div className="flex items-center justify-center h-full bg-muted/20 rounded-md">
      {widget.settings?.compact ? (
        <div className="text-sm text-center text-muted-foreground">Watchlist Widget (Compact)</div>
      ) : (
        <div className="text-center text-muted-foreground">Watchlist Widget</div>
      )}
    </div>
  ),
  news: ({ widget }) => (
    <div className="flex items-center justify-center h-full bg-muted/20 rounded-md">
      {widget.settings?.compact ? (
        <div className="text-sm text-center text-muted-foreground">News Widget (Compact)</div>
      ) : (
        <div className="text-center text-muted-foreground">News Widget</div>
      )}
    </div>
  ),
  performance: ({ widget }) => (
    <div className="flex items-center justify-center h-full bg-muted/20 rounded-md">
      {widget.settings?.compact ? (
        <div className="text-sm text-center text-muted-foreground">Performance Widget (Compact)</div>
      ) : (
        <div className="text-center text-muted-foreground">Performance Widget</div>
      )}
    </div>
  ),
};

// Default widget layouts
const defaultLayout: DashboardLayout = {
  id: 'default',
  name: 'Default Layout',
  widgets: [
    { id: 'chart1', type: 'chart', title: 'BTC/USD Chart', width: 12, height: 2, position: { x: 0, y: 0 } },
    { id: 'portfolio1', type: 'portfolio', title: 'Portfolio Summary', width: 6, height: 1, position: { x: 0, y: 2 } },
    { id: 'positions1', type: 'positions', title: 'Open Positions', width: 6, height: 1, position: { x: 6, y: 2 } },
    { id: 'trade1', type: 'trade', title: 'Quick Trade', width: 4, height: 1, position: { x: 0, y: 3 } },
    { id: 'watchlist1', type: 'watchlist', title: 'Watchlist', width: 4, height: 1, position: { x: 4, y: 3 } },
    { id: 'news1', type: 'news', title: 'Market News', width: 4, height: 1, position: { x: 8, y: 3 } },
  ],
};

// Mobile layout with fewer, stacked widgets
const mobileLayout: DashboardLayout = {
  id: 'mobile',
  name: 'Mobile Layout',
  widgets: [
    { id: 'chart1', type: 'chart', title: 'BTC/USD Chart', width: 12, height: 1, position: { x: 0, y: 0 } },
    { id: 'portfolio1', type: 'portfolio', title: 'Portfolio Summary', width: 12, height: 1, position: { x: 0, y: 1 } },
    { id: 'positions1', type: 'positions', title: 'Open Positions', width: 12, height: 1, position: { x: 0, y: 2 } },
    { id: 'trade1', type: 'trade', title: 'Quick Trade', width: 12, height: 1, position: { x: 0, y: 3 } },
  ],
};

interface ConfigurableDashboardProps {
  className?: string;
}

const ConfigurableDashboard: React.FC<ConfigurableDashboardProps> = ({ className }) => {
  const { isMobile } = useBreakpoints();
  const [layout, setLayout] = useState<DashboardLayout>(defaultLayout);
  const [editMode, setEditMode] = useState(false);
  const [draggedWidget, setDraggedWidget] = useState<string | null>(null);

  // Load the appropriate layout based on screen size
  useEffect(() => {
    if (isMobile) {
      setLayout(mobileLayout);
    } else {
      // Load saved layout from localStorage if available
      const savedLayout = localStorage.getItem('dashboardLayout');
      if (savedLayout) {
        try {
          setLayout(JSON.parse(savedLayout));
        } catch (err) {
          console.error('Error loading saved dashboard layout:', err);
          setLayout(defaultLayout);
        }
      } else {
        setLayout(defaultLayout);
      }
    }
  }, [isMobile]);

  // Save layout to localStorage
  const saveLayout = () => {
    if (!isMobile) { // Don't save mobile layouts
      localStorage.setItem('dashboardLayout', JSON.stringify(layout));
    }
    setEditMode(false);
  };

  // Reset to default layout
  const resetLayout = () => {
    setLayout(isMobile ? mobileLayout : defaultLayout);
    if (!isMobile) {
      localStorage.removeItem('dashboardLayout');
    }
    setEditMode(false);
  };

  // Simulate widget drag start
  const handleDragStart = (widgetId: string) => {
    if (editMode) {
      setDraggedWidget(widgetId);
    }
  };

  // Simulate widget drag end
  const handleDragEnd = () => {
    setDraggedWidget(null);
  };

  // Add a new widget to the dashboard
  const addWidget = (type: WidgetType) => {
    const newWidget: Widget = {
      id: `${type}${Date.now()}`,
      type,
      title: `New ${type.charAt(0).toUpperCase() + type.slice(1)}`,
      width: isMobile ? 12 : 4,
      height: 1,
      position: {
        x: 0,
        y: layout.widgets.length > 0 
          ? Math.max(...layout.widgets.map(w => w.position.y + w.height)) 
          : 0
      }
    };
    
    setLayout({
      ...layout,
      widgets: [...layout.widgets, newWidget]
    });
  };

  // Remove a widget from the dashboard
  const removeWidget = (widgetId: string) => {
    setLayout({
      ...layout,
      widgets: layout.widgets.filter(w => w.id !== widgetId)
    });
  };

  return (
    <div className={cn("flex flex-col", className)}>
      {/* Dashboard Controls */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Trading Dashboard</h2>
        <div className="flex items-center gap-2">
          {editMode ? (
            <>
              <Button size="sm" variant="outline" onClick={resetLayout}>
                <Undo className="w-4 h-4 mr-2" />
                Reset
              </Button>
              <Button size="sm" onClick={saveLayout}>
                <Save className="w-4 h-4 mr-2" />
                Save Layout
              </Button>
            </>
          ) : (
            <Button 
              size="sm" 
              variant="outline" 
              onClick={() => setEditMode(true)}
              disabled={isMobile}
            >
              <Settings className="w-4 h-4 mr-2" />
              Edit Layout
            </Button>
          )}
        </div>
      </div>

      {/* Editable Dashboard Grid */}
      <div 
        className={cn(
          "grid gap-4 transition-all duration-300",
          isMobile ? "grid-cols-1" : "grid-cols-12",
          editMode && "bg-muted/20 p-4 rounded-lg border border-dashed"
        )}
      >
        {layout.widgets.map((widget) => {
          const WidgetComponent = WidgetContent[widget.type];
          const isBeingDragged = draggedWidget === widget.id;
          
          // Calculate the span based on the widget size
          // Since we're not using react-grid-layout, we'll simulate it with CSS grid
          const colSpan = isMobile ? 1 : widget.width;
          const rowSpan = widget.height;
          
          return (
            <Card 
              key={widget.id}
              className={cn(
                "overflow-hidden transition-all duration-200",
                !isMobile && `col-span-${colSpan}`,
                editMode && "border-2 cursor-move",
                isBeingDragged && "border-primary opacity-50",
                editMode && "hover:shadow-md"
              )}
              style={{
                gridRow: `span ${rowSpan}`,
              }}
              draggable={editMode}
              onDragStart={() => handleDragStart(widget.id)}
              onDragEnd={handleDragEnd}
            >
              <CardHeader className="p-3 flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-medium flex items-center">
                  {editMode && <GripVertical className="w-4 h-4 mr-2 text-muted-foreground" />}
                  {widget.title}
                </CardTitle>
                {editMode && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => removeWidget(widget.id)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
              </CardHeader>
              <CardContent className="p-3 pt-0">
                <WidgetComponent widget={widget} compact={isMobile} />
              </CardContent>
            </Card>
          );
        })}
        
        {/* Add Widget Button - Only visible in edit mode */}
        {editMode && !isMobile && (
          <Card className="col-span-12 border-2 border-dashed bg-muted/20 flex items-center justify-center p-4">
            <div className="flex flex-col items-center gap-2">
              <div className="text-muted-foreground mb-2">Add Widgets</div>
              <div className="flex flex-wrap gap-2 justify-center">
                {Object.keys(WidgetContent).map((type) => (
                  <Button 
                    key={type}
                    size="sm"
                    variant="outline"
                    onClick={() => addWidget(type as WidgetType)}
                    className="flex items-center"
                  >
                    <Plus className="w-3 h-3 mr-1" />
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </Button>
                ))}
              </div>
            </div>
          </Card>
        )}
      </div>

      {/* Mobile message about layout customization */}
      {isMobile && (
        <div className="mt-4 p-3 bg-muted/20 rounded-md text-xs text-center text-muted-foreground">
          Please use desktop view to customize your dashboard layout.
        </div>
      )}
    </div>
  );
};

export default ConfigurableDashboard; 