/**
 * Bot Management Service
 * Handles bot configuration, lifecycle, and monitoring
 */
import { RiskSettings } from '../../../generated/prisma';
export interface BotStatus {
    isRunning: boolean;
    lastUpdate: string;
    lastPing?: string;
    health: 'excellent' | 'good' | 'degraded' | 'poor' | 'critical' | 'unknown';
    metrics: {
        tradesExecuted: number;
        profitLoss: number;
        successRate: number;
        latency: number;
    };
    activePositions: number;
    errors: {
        timestamp: string;
        message: string;
        code?: string;
    }[];
    logs: {
        timestamp: string;
        level: 'info' | 'warning' | 'error';
        message: string;
    }[];
}
/**
 * Create a new trading bot
 */
export declare const createBot: (userId: string, botData: {
    name: string;
    symbol: string;
    strategy: string;
    timeframe: string;
    parameters?: Record<string, any>;
}) => Promise<{
    symbol: string;
    userId: string;
    name: string;
    id: string;
    createdAt: Date;
    updatedAt: Date;
    strategy: string;
    timeframe: string;
    parameters: import("../../../generated/prisma/runtime/library").JsonValue;
    isActive: boolean;
}>;
/**
 * Get all bots for a user
 */
export declare const getBotsByUser: (userId: string) => Promise<({
    riskSettings: {
        userId: string;
        name: string;
        id: string;
        createdAt: Date;
        updatedAt: Date;
        isActive: boolean;
        botId: string | null;
        description: string | null;
        positionSizingMethod: string;
        riskPercentage: number;
        maxPositionSize: number;
        kellyFraction: number | null;
        winRate: number | null;
        customSizingParams: import("../../../generated/prisma/runtime/library").JsonValue | null;
        stopLossType: string;
        stopLossValue: number;
        trailingCallback: number | null;
        trailingStep: number | null;
        timeLimit: number | null;
        stopLossLevels: import("../../../generated/prisma/runtime/library").JsonValue | null;
        takeProfitType: string;
        takeProfitValue: number;
        trailingActivation: number | null;
        takeProfitLevels: import("../../../generated/prisma/runtime/library").JsonValue | null;
        maxRiskPerTrade: number;
        maxRiskPerSymbol: number;
        maxRiskPerDirection: number;
        maxTotalRisk: number;
        maxDrawdown: number;
        maxPositions: number;
        maxDailyLoss: number;
        cooldownPeriod: number;
        volatilityLookback: number;
        circuitBreakerEnabled: boolean;
        maxDailyLossBreaker: number;
        maxDrawdownBreaker: number;
        volatilityMultiplier: number;
        consecutiveLossesBreaker: number;
        tradingPause: number;
        marketWideEnabled: boolean;
        enableManualOverride: boolean;
    }[];
} & {
    symbol: string;
    userId: string;
    name: string;
    id: string;
    createdAt: Date;
    updatedAt: Date;
    strategy: string;
    timeframe: string;
    parameters: import("../../../generated/prisma/runtime/library").JsonValue;
    isActive: boolean;
})[]>;
/**
 * Get a specific bot by ID
 */
export declare const getBotById: (botId: string, userId: string) => Promise<{
    riskSettings: {
        userId: string;
        name: string;
        id: string;
        createdAt: Date;
        updatedAt: Date;
        isActive: boolean;
        botId: string | null;
        description: string | null;
        positionSizingMethod: string;
        riskPercentage: number;
        maxPositionSize: number;
        kellyFraction: number | null;
        winRate: number | null;
        customSizingParams: import("../../../generated/prisma/runtime/library").JsonValue | null;
        stopLossType: string;
        stopLossValue: number;
        trailingCallback: number | null;
        trailingStep: number | null;
        timeLimit: number | null;
        stopLossLevels: import("../../../generated/prisma/runtime/library").JsonValue | null;
        takeProfitType: string;
        takeProfitValue: number;
        trailingActivation: number | null;
        takeProfitLevels: import("../../../generated/prisma/runtime/library").JsonValue | null;
        maxRiskPerTrade: number;
        maxRiskPerSymbol: number;
        maxRiskPerDirection: number;
        maxTotalRisk: number;
        maxDrawdown: number;
        maxPositions: number;
        maxDailyLoss: number;
        cooldownPeriod: number;
        volatilityLookback: number;
        circuitBreakerEnabled: boolean;
        maxDailyLossBreaker: number;
        maxDrawdownBreaker: number;
        volatilityMultiplier: number;
        consecutiveLossesBreaker: number;
        tradingPause: number;
        marketWideEnabled: boolean;
        enableManualOverride: boolean;
    }[];
} & {
    symbol: string;
    userId: string;
    name: string;
    id: string;
    createdAt: Date;
    updatedAt: Date;
    strategy: string;
    timeframe: string;
    parameters: import("../../../generated/prisma/runtime/library").JsonValue;
    isActive: boolean;
}>;
/**
 * Update a bot's configuration
 */
export declare const updateBot: (botId: string, userId: string, updateData: {
    name?: string;
    symbol?: string;
    strategy?: string;
    timeframe?: string;
    parameters?: Record<string, any>;
}) => Promise<{
    symbol: string;
    userId: string;
    name: string;
    id: string;
    createdAt: Date;
    updatedAt: Date;
    strategy: string;
    timeframe: string;
    parameters: import("../../../generated/prisma/runtime/library").JsonValue;
    isActive: boolean;
}>;
/**
 * Delete a bot
 */
export declare const deleteBot: (botId: string, userId: string) => Promise<boolean>;
/**
 * Start a bot
 */
export declare const startBot: (botId: string, userId: string) => Promise<boolean>;
/**
 * Stop a bot
 */
export declare const stopBot: (botId: string, userId: string) => Promise<boolean>;
/**
 * Pause a bot (temporarily suspend operations without full stop)
 */
export declare const pauseBot: (botId: string, userId: string, duration?: number) => Promise<boolean>;
/**
 * Get bot status
 */
export declare const getBotStatus: (botId: string, userId: string) => Promise<BotStatus | null>;
/**
 * Configure risk settings for a bot
 */
export declare const configureBotRiskSettings: (botId: string, userId: string, riskConfig: Partial<RiskSettings>) => Promise<{
    userId: string;
    name: string;
    id: string;
    createdAt: Date;
    updatedAt: Date;
    isActive: boolean;
    botId: string | null;
    description: string | null;
    positionSizingMethod: string;
    riskPercentage: number;
    maxPositionSize: number;
    kellyFraction: number | null;
    winRate: number | null;
    customSizingParams: import("../../../generated/prisma/runtime/library").JsonValue | null;
    stopLossType: string;
    stopLossValue: number;
    trailingCallback: number | null;
    trailingStep: number | null;
    timeLimit: number | null;
    stopLossLevels: import("../../../generated/prisma/runtime/library").JsonValue | null;
    takeProfitType: string;
    takeProfitValue: number;
    trailingActivation: number | null;
    takeProfitLevels: import("../../../generated/prisma/runtime/library").JsonValue | null;
    maxRiskPerTrade: number;
    maxRiskPerSymbol: number;
    maxRiskPerDirection: number;
    maxTotalRisk: number;
    maxDrawdown: number;
    maxPositions: number;
    maxDailyLoss: number;
    cooldownPeriod: number;
    volatilityLookback: number;
    circuitBreakerEnabled: boolean;
    maxDailyLossBreaker: number;
    maxDrawdownBreaker: number;
    volatilityMultiplier: number;
    consecutiveLossesBreaker: number;
    tradingPause: number;
    marketWideEnabled: boolean;
    enableManualOverride: boolean;
}>;
/**
 * Update bot health status
 * This function would be called by a health monitoring service
 */
export declare const updateBotHealth: (botId: string, healthData: {
    health: BotStatus["health"];
    metrics: Partial<BotStatus["metrics"]>;
    errors?: BotStatus["errors"];
    logs?: BotStatus["logs"];
}) => Promise<boolean>;
