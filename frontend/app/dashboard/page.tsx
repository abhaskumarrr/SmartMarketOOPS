/**
 * SmartMarketOOPS Dashboard Page
 * Main entry point for the Real-Time Trading Dashboard with ML Intelligence
 */

'use client';

import React from 'react';
import { UnifiedPageWrapper } from '../../components/layout/UnifiedPageWrapper';
import { RealTimeTradingDashboard } from '../../components/trading/RealTimeTradingDashboard';

export default function DashboardPage() {
  return (
    <UnifiedPageWrapper connectionStatus="connected" showConnectionStatus={true}>
      <RealTimeTradingDashboard defaultSymbol="BTCUSD" />
    </UnifiedPageWrapper>
  );
}
