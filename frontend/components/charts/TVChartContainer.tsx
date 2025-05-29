import React, { useEffect, useRef, useState } from 'react';
import {
  createChart,
  IChartApi,
  DeepPartial,
  ChartOptions,
  LineData,
  ColorType,
} from 'lightweight-charts';
import useWebSocket from '../../lib/hooks/useWebSocket';

interface TVChartContainerProps {
  symbol: string;
  interval: string;
  data: Array<{ time: number | string; value: number }>;
  signals?: Array<{ time: number | string; type: 'buy' | 'sell' | 'hold'; confidence?: number }>;
}

export const TVChartContainer: React.FC<TVChartContainerProps> = ({
  symbol,
  interval,
  data: initialData,
  signals,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [chartData, setChartData] = useState(initialData);
  
  // Use our WebSocket hook instead of direct socket.io connection
  const { data: wsData, status, isConnected } = useWebSocket<any>(
    'market',
    'market:data',
    symbol
  );

  // Update chart data when new WebSocket data arrives
  useEffect(() => {
    if (wsData && wsData.symbol === symbol && wsData.data) {
      setChartData((prev) => {
        // Merge or replace logic as needed
        if (Array.isArray(wsData.data)) return wsData.data;
        if (wsData.data.time && wsData.data.value) return [...prev, wsData.data];
        return prev;
      });
    }
  }, [wsData, symbol]);

  useEffect(() => {
    if (!chartContainerRef.current) return;
    if (chartRef.current) {
      chartRef.current.remove();
    }
    const chartOptions: DeepPartial<ChartOptions> = {
      layout: {
        background: { type: ColorType.Solid, color: '#181A20' },
        textColor: '#D9D9D9',
      },
      grid: {
        vertLines: { color: '#222' },
        horzLines: { color: '#222' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 500,
      timeScale: { timeVisible: true, secondsVisible: false },
    };
    const chart = createChart(chartContainerRef.current, chartOptions);
    chartRef.current = chart;
    // @ts-expect-error: addLineSeries is available at runtime in lightweight-charts v5
    const lineSeries = chart.addLineSeries({ color: '#2962FF', lineWidth: 2 });
    lineSeries.setData(chartData as LineData[]);
    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current!.clientWidth });
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [symbol, interval, chartData]);

  return <div style={{ width: '100%', height: 500 }} ref={chartContainerRef} />;
};
 