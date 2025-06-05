/**
 * Data Consistency Verification Script
 * Verifies that all trading operations use consistent live data sources
 */
interface DataConsistencyReport {
    timestamp: string;
    marketDataProvider: {
        name: string;
        isLive: boolean;
        isMock: boolean;
    };
    deltaExchangeService: {
        isReady: boolean;
        testnetMode: boolean;
    };
    consistencyChecks: {
        dataSourceUnified: boolean;
        noMockDataInProduction: boolean;
        deltaServiceReady: boolean;
        priceDataConsistent: boolean;
    };
    testResults: {
        marketDataFetch: boolean;
        deltaExchangeDataFetch: boolean;
        priceComparison: {
            marketDataPrice: number;
            deltaExchangePrice: number;
            difference: number;
            withinTolerance: boolean;
        };
    };
    recommendations: string[];
    overallStatus: 'SAFE' | 'WARNING' | 'CRITICAL';
}
declare function verifyDataConsistency(): Promise<DataConsistencyReport>;
declare function testTradingBotDataConsistency(): Promise<void>;
export { verifyDataConsistency, testTradingBotDataConsistency };
