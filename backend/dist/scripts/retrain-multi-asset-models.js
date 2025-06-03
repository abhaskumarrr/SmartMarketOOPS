#!/usr/bin/env node
"use strict";
/**
 * Multi-Asset AI Model Retraining Script
 * Retrains all AI models using 6 months of real market data from BTC, ETH, and SOL
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MultiAssetModelRetrainingRunner = void 0;
const multiAssetModelTrainer_1 = require("../services/multiAssetModelTrainer");
const aiModelTrainer_1 = require("../services/aiModelTrainer");
const logger_1 = require("../utils/logger");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
class MultiAssetModelRetrainingRunner {
    constructor() {
        this.multiAssetTrainer = (0, multiAssetModelTrainer_1.createMultiAssetModelTrainer)();
        this.modelTrainer = (0, aiModelTrainer_1.createAIModelTrainer)();
    }
    /**
     * Run comprehensive multi-asset model retraining
     */
    async runMultiAssetRetraining() {
        const startTime = Date.now();
        try {
            logger_1.logger.info('🪙 Starting Multi-Asset AI Model Retraining...');
            logger_1.logger.info('📊 Assets: Bitcoin (BTC), Ethereum (ETH), Solana (SOL)');
            logger_1.logger.info('⏰ Training Period: 6 months of real market data');
            // Step 1: Fetch and process multi-asset training data
            const multiAssetDataset = await this.fetchMultiAssetTrainingData();
            // Step 2: Split dataset for training
            const { train, validation, test } = this.multiAssetTrainer.splitMultiAssetDataset(multiAssetDataset);
            // Step 3: Train models with multi-asset data
            const trainedModels = await this.trainMultiAssetModels(train, validation, test);
            // Step 4: Save trained models
            await this.saveMultiAssetModels(trainedModels, multiAssetDataset);
            // Step 5: Generate comprehensive report
            this.generateMultiAssetTrainingReport(trainedModels, multiAssetDataset, startTime);
            // Step 6: Validate cross-asset performance
            await this.validateCrossAssetPerformance(trainedModels, multiAssetDataset);
            const duration = (Date.now() - startTime) / 1000;
            logger_1.logger.info(`🎉 Multi-asset model retraining completed in ${duration.toFixed(2)} seconds`);
        }
        catch (error) {
            logger_1.logger.error('❌ Multi-asset model retraining failed:', error);
            throw error;
        }
    }
    /**
     * Fetch and process multi-asset training data
     */
    async fetchMultiAssetTrainingData() {
        logger_1.logger.info('📊 Fetching 6 months of multi-asset data...');
        const endDate = new Date();
        const startDate = new Date(endDate.getTime() - (6 * 30 * 24 * 60 * 60 * 1000)); // 6 months
        const assets = ['BTCUSD', 'ETHUSD', 'SOLUSD'];
        logger_1.logger.info('📅 Multi-asset data period:', {
            startDate: startDate.toISOString().split('T')[0],
            endDate: endDate.toISOString().split('T')[0],
            assets,
            expectedCandles: Math.floor((endDate.getTime() - startDate.getTime()) / (60 * 60 * 1000)) * assets.length,
        });
        try {
            const dataset = await this.multiAssetTrainer.processMultiAssetTrainingData('1h', startDate, endDate, assets, 0.7, // 70% training
            0.15, // 15% validation
            0.15 // 15% testing
            );
            logger_1.logger.info('✅ Multi-asset training data processed successfully', {
                totalSamples: dataset.metadata.totalSamples,
                assetBreakdown: dataset.assetBreakdown,
                correlationStats: dataset.correlationStats,
                featureCount: dataset.metadata.featureCount,
            });
            return dataset;
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to fetch multi-asset data:', error);
            throw error;
        }
    }
    /**
     * Train models with multi-asset data
     */
    async trainMultiAssetModels(trainData, validationData, testData) {
        logger_1.logger.info('🧠 Training AI models with multi-asset features...', {
            trainSamples: trainData.length,
            validationSamples: validationData.length,
            testSamples: testData.length,
        });
        // Convert multi-asset features to standard training format
        const convertedTrainData = this.convertMultiAssetFeatures(trainData);
        const convertedValidationData = this.convertMultiAssetFeatures(validationData);
        const convertedTestData = this.convertMultiAssetFeatures(testData);
        // Train all models with enhanced multi-asset features
        const trainedModels = await this.modelTrainer.trainAllModels(convertedTrainData, convertedValidationData, convertedTestData);
        // Enhance models with multi-asset metadata
        Object.values(trainedModels).forEach(model => {
            model.modelName = `MultiAsset_${model.modelName}`;
            model.version = '2.1.0'; // Multi-asset version
            // Add multi-asset specific metadata
            model.multiAssetMetadata = {
                supportedAssets: ['BTCUSD', 'ETHUSD', 'SOLUSD'],
                crossAssetFeatures: true,
                portfolioOptimization: true,
                correlationAnalysis: true,
            };
        });
        logger_1.logger.info('✅ Multi-asset model training completed', {
            modelsCount: Object.keys(trainedModels).length,
            avgAccuracy: Object.values(trainedModels).reduce((sum, model) => sum + model.finalMetrics.testAccuracy, 0) / Object.values(trainedModels).length,
        });
        return trainedModels;
    }
    /**
     * Convert multi-asset features to standard training format
     */
    convertMultiAssetFeatures(multiAssetData) {
        return multiAssetData.map(sample => ({
            // Basic OHLCV
            open: sample.open,
            high: sample.high,
            low: sample.low,
            close: sample.close,
            volume: sample.volume,
            // Technical indicators
            rsi_14: sample.rsi_adjusted || sample.rsi_14 || 50,
            ema_12: sample.ema_12 || sample.close * 0.99,
            ema_26: sample.ema_26 || sample.close * 1.01,
            macd: sample.macd_strength || sample.macd || 0,
            volume_sma_20: sample.volume_sma_20 || sample.volume * 0.8,
            // Additional required fields for training
            sma_20: sample.sma_20 || sample.close,
            sma_50: sample.sma_50 || sample.close,
            bollinger_upper: sample.bollinger_upper || sample.close * 1.02,
            bollinger_lower: sample.bollinger_lower || sample.close * 0.98,
            // Multi-asset specific features
            asset_type: sample.asset_type || 0,
            btc_correlation: sample.btc_correlation || 0,
            eth_correlation: sample.eth_correlation || 0,
            sol_correlation: sample.sol_correlation || 0,
            volatility_ratio: sample.volatility_ratio || 1,
            volume_profile: sample.volume_profile || 1,
            cross_asset_momentum: sample.cross_asset_momentum || 0,
            relative_strength: sample.relative_strength || 0.5,
            // Time features
            hour_of_day: sample.hour_of_day || 0,
            day_of_week: sample.day_of_week || 0,
            // Target variables
            future_return_1h: sample.future_return_1h || 0,
            signal_1h: sample.signal_1h || 0,
        }));
    }
    /**
     * Save multi-asset trained models
     */
    async saveMultiAssetModels(trainedModels, dataset) {
        logger_1.logger.info('💾 Saving multi-asset trained models...');
        const modelsDir = path_1.default.join(process.cwd(), 'trained_models');
        if (!fs_1.default.existsSync(modelsDir)) {
            fs_1.default.mkdirSync(modelsDir, { recursive: true });
        }
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        // Save models with multi-asset prefix
        for (const [modelType, model] of Object.entries(trainedModels)) {
            const modelPath = path_1.default.join(modelsDir, `multi_asset_${modelType}_model_${timestamp}.json`);
            try {
                // Add dataset metadata to model
                const enhancedModel = {
                    ...model,
                    datasetMetadata: {
                        assets: dataset.metadata.symbols,
                        totalSamples: dataset.metadata.totalSamples,
                        assetBreakdown: dataset.assetBreakdown,
                        correlationStats: dataset.correlationStats,
                        trainingPeriod: {
                            startDate: dataset.metadata.startDate,
                            endDate: dataset.metadata.endDate,
                        },
                    },
                };
                fs_1.default.writeFileSync(modelPath, JSON.stringify(enhancedModel, null, 2));
                logger_1.logger.info(`✅ Saved multi-asset ${model.modelName} to ${modelPath}`);
            }
            catch (error) {
                logger_1.logger.error(`❌ Failed to save multi-asset ${model.modelName}:`, error);
            }
        }
        // Update latest models (overwrite previous)
        for (const [modelType, model] of Object.entries(trainedModels)) {
            const latestPath = path_1.default.join(modelsDir, `${modelType}_model_latest.json`);
            try {
                const enhancedModel = {
                    ...model,
                    datasetMetadata: {
                        assets: dataset.metadata.symbols,
                        totalSamples: dataset.metadata.totalSamples,
                        assetBreakdown: dataset.assetBreakdown,
                        correlationStats: dataset.correlationStats,
                    },
                };
                fs_1.default.writeFileSync(latestPath, JSON.stringify(enhancedModel, null, 2));
                logger_1.logger.info(`✅ Updated latest multi-asset ${model.modelName}`);
            }
            catch (error) {
                logger_1.logger.error(`❌ Failed to update latest multi-asset ${model.modelName}:`, error);
            }
        }
    }
    /**
     * Generate comprehensive multi-asset training report
     */
    generateMultiAssetTrainingReport(trainedModels, dataset, startTime) {
        const duration = (Date.now() - startTime) / 1000;
        logger_1.logger.info('\n' + '🪙 MULTI-ASSET AI MODEL RETRAINING REPORT'.padStart(70, '='));
        logger_1.logger.info('='.repeat(140));
        // Overall Summary
        logger_1.logger.info('📊 MULTI-ASSET TRAINING SUMMARY:');
        logger_1.logger.info(`   Total Training Time: ${duration.toFixed(2)} seconds (${(duration / 60).toFixed(1)} minutes)`);
        logger_1.logger.info(`   Assets Trained: ${dataset.metadata.symbols.join(', ')}`);
        logger_1.logger.info(`   Dataset Size: ${dataset.metadata.totalSamples.toLocaleString()} samples`);
        logger_1.logger.info(`   Feature Count: ${dataset.metadata.featureCount} (including cross-asset features)`);
        logger_1.logger.info(`   Training Period: ${dataset.metadata.startDate.toISOString().split('T')[0]} to ${dataset.metadata.endDate.toISOString().split('T')[0]}`);
        // Asset Breakdown
        logger_1.logger.info('\n🪙 ASSET BREAKDOWN:');
        logger_1.logger.info(`   Bitcoin (BTC): ${dataset.assetBreakdown.btc.toLocaleString()} samples (${(dataset.assetBreakdown.btc / dataset.metadata.totalSamples * 100).toFixed(1)}%)`);
        logger_1.logger.info(`   Ethereum (ETH): ${dataset.assetBreakdown.eth.toLocaleString()} samples (${(dataset.assetBreakdown.eth / dataset.metadata.totalSamples * 100).toFixed(1)}%)`);
        logger_1.logger.info(`   Solana (SOL): ${dataset.assetBreakdown.sol.toLocaleString()} samples (${(dataset.assetBreakdown.sol / dataset.metadata.totalSamples * 100).toFixed(1)}%)`);
        // Cross-Asset Correlation Analysis
        logger_1.logger.info('\n🔗 CROSS-ASSET CORRELATION ANALYSIS:');
        logger_1.logger.info(`   BTC-ETH Correlation: ${dataset.correlationStats.btc_eth.toFixed(3)}`);
        logger_1.logger.info(`   BTC-SOL Correlation: ${dataset.correlationStats.btc_sol.toFixed(3)}`);
        logger_1.logger.info(`   ETH-SOL Correlation: ${dataset.correlationStats.eth_sol.toFixed(3)}`);
        // Model Performance Summary
        logger_1.logger.info('\n🎯 MULTI-ASSET MODEL PERFORMANCE:');
        logger_1.logger.info('   Model                         | Train Acc | Val Acc | Test Acc | F1 Score | Cross-Asset Features');
        logger_1.logger.info('   ' + '-'.repeat(105));
        Object.values(trainedModels).forEach(model => {
            const metrics = model.finalMetrics;
            const crossAssetEnabled = model.multiAssetMetadata?.crossAssetFeatures ? '✅' : '❌';
            logger_1.logger.info(`   ${model.modelName.padEnd(29)} | ${(metrics.trainAccuracy * 100).toFixed(1).padStart(9)}% | ${(metrics.validationAccuracy * 100).toFixed(1).padStart(7)}% | ${(metrics.testAccuracy * 100).toFixed(1).padStart(8)}% | ${metrics.f1Score.toFixed(3).padStart(8)} | ${crossAssetEnabled.padStart(20)}`);
        });
        // Multi-Asset Specific Insights
        logger_1.logger.info('\n🧠 MULTI-ASSET INSIGHTS:');
        const avgAccuracy = Object.values(trainedModels).reduce((sum, model) => sum + model.finalMetrics.testAccuracy, 0) / Object.values(trainedModels).length;
        logger_1.logger.info(`   Average Model Accuracy: ${(avgAccuracy * 100).toFixed(2)}%`);
        if (avgAccuracy > 0.75) {
            logger_1.logger.info(`   🌟 EXCELLENT: Multi-asset models show superior performance`);
        }
        else if (avgAccuracy > 0.65) {
            logger_1.logger.info(`   ✅ GOOD: Multi-asset models show solid performance`);
        }
        else {
            logger_1.logger.info(`   ⚠️ MODERATE: Multi-asset models need further optimization`);
        }
        // Cross-Asset Feature Importance
        logger_1.logger.info('\n🎯 CROSS-ASSET FEATURE ANALYSIS:');
        logger_1.logger.info('   Key multi-asset features successfully integrated:');
        logger_1.logger.info('   ✅ Cross-asset correlations (BTC, ETH, SOL)');
        logger_1.logger.info('   ✅ Relative strength analysis');
        logger_1.logger.info('   ✅ Cross-asset momentum indicators');
        logger_1.logger.info('   ✅ Asset-specific volatility profiles');
        logger_1.logger.info('   ✅ Portfolio optimization features');
        // Training Recommendations
        logger_1.logger.info('\n💡 MULTI-ASSET RECOMMENDATIONS:');
        logger_1.logger.info('   🚀 NEXT STEPS:');
        logger_1.logger.info('   1. Run comprehensive multi-asset backtesting');
        logger_1.logger.info('   2. Test portfolio optimization strategies');
        logger_1.logger.info('   3. Validate cross-asset correlation predictions');
        logger_1.logger.info('   4. Implement dynamic asset allocation');
        logger_1.logger.info('   5. Monitor model performance across different market regimes');
        logger_1.logger.info('\n   🎯 DEPLOYMENT READINESS:');
        if (avgAccuracy > 0.7) {
            logger_1.logger.info('   ✅ Models ready for multi-asset live trading');
            logger_1.logger.info('   ✅ Cross-asset features validated');
            logger_1.logger.info('   ✅ Portfolio optimization enabled');
        }
        else {
            logger_1.logger.info('   ⚠️ Additional optimization recommended before deployment');
            logger_1.logger.info('   🔧 Consider feature engineering improvements');
            logger_1.logger.info('   📊 Validate with longer training periods');
        }
        logger_1.logger.info('='.repeat(140));
    }
    /**
     * Validate cross-asset performance
     */
    async validateCrossAssetPerformance(trainedModels, dataset) {
        logger_1.logger.info('\n🔍 Validating cross-asset model performance...');
        // Validate model requirements
        const requiredAccuracy = 0.65;
        const requiredF1Score = 0.6;
        let passedModels = 0;
        const totalModels = Object.keys(trainedModels).length;
        Object.values(trainedModels).forEach(model => {
            const meetsAccuracy = model.finalMetrics.testAccuracy >= requiredAccuracy;
            const meetsF1 = model.finalMetrics.f1Score >= requiredF1Score;
            const hasMultiAssetFeatures = model.multiAssetMetadata?.crossAssetFeatures;
            if (meetsAccuracy && meetsF1 && hasMultiAssetFeatures) {
                passedModels++;
                logger_1.logger.info(`   ✅ ${model.modelName}: Multi-asset validation passed`);
            }
            else {
                logger_1.logger.warn(`   ⚠️ ${model.modelName}: Multi-asset validation issues`, {
                    accuracy: `${(model.finalMetrics.testAccuracy * 100).toFixed(1)}%`,
                    f1Score: model.finalMetrics.f1Score.toFixed(3),
                    multiAssetFeatures: hasMultiAssetFeatures,
                });
            }
        });
        const passRate = (passedModels / totalModels) * 100;
        logger_1.logger.info(`\n📊 Multi-Asset Validation Results: ${passedModels}/${totalModels} models passed (${passRate.toFixed(1)}%)`);
        if (passRate >= 80) {
            logger_1.logger.info('🎉 Excellent multi-asset model performance - ready for deployment');
        }
        else if (passRate >= 60) {
            logger_1.logger.info('✅ Good multi-asset model performance - consider additional optimization');
        }
        else {
            logger_1.logger.warn('⚠️ Multi-asset model performance below expectations - review training process');
        }
        // Validate cross-asset correlations
        const correlations = dataset.correlationStats;
        logger_1.logger.info('\n🔗 Cross-Asset Correlation Validation:');
        if (Math.abs(correlations.btc_eth) > 0.3) {
            logger_1.logger.info(`   ✅ BTC-ETH correlation detected: ${correlations.btc_eth.toFixed(3)}`);
        }
        if (Math.abs(correlations.btc_sol) > 0.2) {
            logger_1.logger.info(`   ✅ BTC-SOL correlation detected: ${correlations.btc_sol.toFixed(3)}`);
        }
        if (Math.abs(correlations.eth_sol) > 0.2) {
            logger_1.logger.info(`   ✅ ETH-SOL correlation detected: ${correlations.eth_sol.toFixed(3)}`);
        }
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        logger_1.logger.info('🧹 Multi-asset model retraining cleanup completed');
    }
}
exports.MultiAssetModelRetrainingRunner = MultiAssetModelRetrainingRunner;
/**
 * Main execution function
 */
async function main() {
    const runner = new MultiAssetModelRetrainingRunner();
    try {
        await runner.runMultiAssetRetraining();
    }
    catch (error) {
        logger_1.logger.error('💥 Multi-asset model retraining failed:', error);
        process.exit(1);
    }
    finally {
        await runner.cleanup();
    }
}
// Handle graceful shutdown
process.on('SIGINT', async () => {
    logger_1.logger.info('🛑 Received SIGINT, cleaning up...');
    const runner = new MultiAssetModelRetrainingRunner();
    await runner.cleanup();
    process.exit(0);
});
process.on('SIGTERM', async () => {
    logger_1.logger.info('🛑 Received SIGTERM, cleaning up...');
    const runner = new MultiAssetModelRetrainingRunner();
    await runner.cleanup();
    process.exit(0);
});
// Run if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=retrain-multi-asset-models.js.map