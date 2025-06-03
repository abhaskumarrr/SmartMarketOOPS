/**
 * Bot Management Page
 * Main page for bot management functionality
 */

'use client';

import React from 'react';
import { UnifiedPageWrapper } from '../../components/layout/UnifiedPageWrapper';
import { BotManagementDashboard } from '../../components/bots/BotManagementDashboard';

export default function BotsPage() {
  return (
    <UnifiedPageWrapper connectionStatus="connected" showConnectionStatus={true}>
      <BotManagementDashboard />
    </UnifiedPageWrapper>
  );
}
