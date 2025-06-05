/**
 * Delta Exchange Trading Page
 * Professional trading bot management interface for Delta Exchange India testnet
 */

'use client';

import React from 'react';
import { UnifiedPageWrapper } from '../../components/layout/UnifiedPageWrapper';
import { DeltaTradingDashboard } from '../../components/trading/DeltaTradingDashboard';

export default function DeltaTradingPage() {
  return (
    <UnifiedPageWrapper 
      connectionStatus="connected" 
      showConnectionStatus={true}
      fullHeight={true}
      disablePadding={true}
    >
      <DeltaTradingDashboard />
    </UnifiedPageWrapper>
  );
}
