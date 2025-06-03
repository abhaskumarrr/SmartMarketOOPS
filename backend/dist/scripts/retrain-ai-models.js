#!/usr/bin/env node
"use strict";
/**
 * AI Model Retraining Script
 * Retrains all AI models using 6 months of real Binance data
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AIModelRetrainingRunner = void 0;
const marketDataProvider_1 = require("../services/marketDataProvider");
const modelTrainingDataProcessor_1 = require("../services/modelTrainingDataProcessor");
const aiModelTrainer_1 = require("../services/aiModelTrainer");
const logger_1 = require("../utils/logger");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
class AIModelRetrainingRunner {
    constructor() {
        this.dataProcessor = (0, modelTrainingDataProcessor_1.createModelTrainingDataProcessor)();
        this.modelTrainer = (0, aiModelTrainer_1.createAIModelTrainer)();
    }
    /**
     * Run comprehensive AI model retraining
     */
    async runRetraining() {
        const startTime = Date.now();
        try {
            logger_1.logger.info('🧠 Starting AI Model Retraining with 6 months of real data...');
            // Step 1: Fetch 6 months of real market data
            const marketData = await this.fetchTrainingData();
            // Step 2: Process data for training
            const trainingDataset = this.dataProcessor.processTrainingData(marketData, 'BTCUSD', 0.7, // 70% training
            0.15, // 15% validation
            0.15 // 15% testing
            );
            // Step 3: Split dataset
            const { train, validation, test } = this.dataProcessor.splitDataset(trainingDataset);
            // Step 4: Train all models
            const trainedModels = await this.modelTrainer.trainAllModels(train, validation, test);
            // Step 5: Save trained models
            await this.saveTrainedModels(trainedModels);
            // Step 6: Generate training report
            this.generateTrainingReport(trainedModels, trainingDataset, startTime);
            // Step 7: Validate model improvements
            await this.validateModelImprovements(trainedModels);
            const duration = (Date.now() - startTime) / 1000;
            logger_1.logger.info(`🎉 AI model retraining completed in ${duration.toFixed(2)} seconds`);
        }
        catch (error) {
            logger_1.logger.error('❌ AI model retraining failed:', error);
            throw error;
        }
    }
    /**
     * Fetch 6 months of real market data from Binance
     */
    async fetchTrainingData() {
        logger_1.logger.info('📊 Fetching 6 months of real market data from Binance...');
        const endDate = new Date();
        const startDate = new Date(endDate.getTime() - (6 * 30 * 24 * 60 * 60 * 1000)); // 6 months
        logger_1.logger.info('📅 Data period:', {
            startDate: startDate.toISOString().split('T')[0],
            endDate: endDate.toISOString().split('T')[0],
            expectedCandles: Math.floor((endDate.getTime() - startDate.getTime()) / (60 * 60 * 1000)), // Hourly candles
        });
        try {
            const response = await marketDataProvider_1.marketDataService.fetchHistoricalData({
                symbol: 'BTCUSD',
                timeframe: '1h',
                startDate,
                endDate,
                exchange: 'binance',
            }, 'binance');
            logger_1.logger.info('✅ Real market data fetched successfully', {
                dataPoints: response.count,
                source: response.source,
                symbol: response.symbol,
                timeframe: response.timeframe,
            });
            if (response.count < 1000) {
                logger_1.logger.warn('⚠️ Limited data available, using available data for training');
            }
            return response.data;
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to fetch real data from Binance:', error);
            logger_1.logger.info('🔄 Falling back to enhanced mock data for training...');
            // Fallback to enhanced mock data
            const response = await marketDataProvider_1.marketDataService.fetchHistoricalData({
                symbol: 'BTCUSD',
                timeframe: '1h',
                startDate,
                endDate,
                exchange: 'enhanced-mock',
            }, 'enhanced-mock');
            logger_1.logger.info('✅ Enhanced mock data generated for training', {
                dataPoints: response.count,
                source: response.source,
            });
            return response.data;
        }
    }
    /**
     * Save trained models to disk
     */
    async saveTrainedModels(trainedModels) {
        logger_1.logger.info('💾 Saving trained models...');
        const modelsDir = path_1.default.join(process.cwd(), 'trained_models');
        if (!fs_1.default.existsSync(modelsDir)) {
            fs_1.default.mkdirSync(modelsDir, { recursive: true });
        }
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        for (const [modelType, model] of Object.entries(trainedModels)) {
            const modelPath = path_1.default.join(modelsDir, `${modelType}_model_${timestamp}.json`);
            try {
                fs_1.default.writeFileSync(modelPath, JSON.stringify(model, null, 2));
                logger_1.logger.info(`✅ Saved ${model.modelName} to ${modelPath}`);
            }
            catch (error) {
                logger_1.logger.error(`❌ Failed to save ${model.modelName}:`, error);
            }
        }
        // Save latest models (overwrite previous)
        for (const [modelType, model] of Object.entries(trainedModels)) {
            const latestPath = path_1.default.join(modelsDir, `${modelType}_model_latest.json`);
            try {
                fs_1.default.writeFileSync(latestPath, JSON.stringify(model, null, 2));
                logger_1.logger.info(`✅ Updated latest ${model.modelName}`);
            }
            catch (error) {
                logger_1.logger.error(`❌ Failed to update latest ${model.modelName}:`, error);
            }
        }
    }
    /**
     * Generate comprehensive training report
     */
    generateTrainingReport(trainedModels, dataset, startTime) {
        const duration = (Date.now() - startTime) / 1000;
        logger_1.logger.info('\n' + '🧠 AI MODEL RETRAINING REPORT'.padStart(60, '='));
        logger_1.logger.info('='.repeat(120));
        // Overall Summary
        logger_1.logger.info('📊 TRAINING SUMMARY:');
        logger_1.logger.info(`   Total Training Time: ${duration.toFixed(2)} seconds (${(duration / 60).toFixed(1)} minutes)`);
        logger_1.logger.info(`   Dataset Size: ${dataset.metadata.totalSamples.toLocaleString()} samples`);
        logger_1.logger.info(`   Feature Count: ${dataset.metadata.featureCount}`);
        logger_1.logger.info(`   Data Period: ${dataset.metadata.startDate.toISOString().split('T')[0]} to ${dataset.metadata.endDate.toISOString().split('T')[0]}`);
        logger_1.logger.info(`   Models Trained: ${Object.keys(trainedModels).length}`);
        // Model Performance Summary
        logger_1.logger.info('\n🎯 MODEL PERFORMANCE SUMMARY:');
        logger_1.logger.info('   Model                    | Train Acc | Val Acc | Test Acc | F1 Score | Training Time');
        logger_1.logger.info('   ' + '-'.repeat(85));
        Object.values(trainedModels).forEach(model => {
            const metrics = model.finalMetrics;
            logger_1.logger.info(`   ${model.modelName.padEnd(24)} | ${(metrics.trainAccuracy * 100).toFixed(1).padStart(9)}% | ${(metrics.validationAccuracy * 100).toFixed(1).padStart(7)}% | ${(metrics.testAccuracy * 100).toFixed(1).padStart(8)}% | ${metrics.f1Score.toFixed(3).padStart(8)} | ${model.trainingTime.toFixed(1).padStart(13)}s`);
        });
        // Individual Model Details
        Object.values(trainedModels).forEach(model => {
            logger_1.logger.info(`\n🔬 ${model.modelName.toUpperCase()} DETAILS:`);
            logger_1.logger.info(`   Version: ${model.version}`);
            logger_1.logger.info(`   Trained At: ${model.trainedAt.toISOString()}`);
            logger_1.logger.info(`   Training Epochs: ${model.trainingHistory.length}`);
            logger_1.logger.info(`   Final Learning Rate: ${model.trainingHistory[model.trainingHistory.length - 1]?.learningRate.toFixed(6)}`);
            logger_1.logger.info(`   📈 Performance Metrics:`);
            logger_1.logger.info(`     Training Accuracy: ${(model.finalMetrics.trainAccuracy * 100).toFixed(2)}%`);
            logger_1.logger.info(`     Validation Accuracy: ${(model.finalMetrics.validationAccuracy * 100).toFixed(2)}%`);
            logger_1.logger.info(`     Test Accuracy: ${(model.finalMetrics.testAccuracy * 100).toFixed(2)}%`);
            logger_1.logger.info(`     Precision: ${model.finalMetrics.precision.toFixed(3)}`);
            logger_1.logger.info(`     Recall: ${model.finalMetrics.recall.toFixed(3)}`);
            logger_1.logger.info(`     F1 Score: ${model.finalMetrics.f1Score.toFixed(3)}`);
            // Top 5 most important features
            const sortedFeatures = Object.entries(model.featureImportance)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 5);
            logger_1.logger.info(`   🎯 Top 5 Important Features:`);
            sortedFeatures.forEach(([feature, importance], index) => {
                logger_1.logger.info(`     ${index + 1}. ${feature}: ${importance.toFixed(3)}`);
            });
            // Training convergence
            const firstEpoch = model.trainingHistory[0];
            const lastEpoch = model.trainingHistory[model.trainingHistory.length - 1];
            const improvement = lastEpoch.validationAccuracy - firstEpoch.validationAccuracy;
            logger_1.logger.info(`   📊 Training Convergence:`);
            logger_1.logger.info(`     Initial Val Accuracy: ${(firstEpoch.validationAccuracy * 100).toFixed(2)}%`);
            logger_1.logger.info(`     Final Val Accuracy: ${(lastEpoch.validationAccuracy * 100).toFixed(2)}%`);
            logger_1.logger.info(`     Total Improvement: ${(improvement * 100).toFixed(2)}%`);
        });
        // Training Recommendations
        logger_1.logger.info('\n💡 TRAINING INSIGHTS:');
        const bestModel = Object.values(trainedModels).reduce((best, current) => current.finalMetrics.testAccuracy > best.finalMetrics.testAccuracy ? current : best);
        logger_1.logger.info(`   🏆 Best Performing Model: ${bestModel.modelName} (${(bestModel.finalMetrics.testAccuracy * 100).toFixed(2)}% test accuracy)`);
        const avgAccuracy = Object.values(trainedModels).reduce((sum, model) => sum + model.finalMetrics.testAccuracy, 0) / Object.values(trainedModels).length;
        logger_1.logger.info(`   📊 Average Model Accuracy: ${(avgAccuracy * 100).toFixed(2)}%`);
        if (avgAccuracy > 0.7) {
            logger_1.logger.info(`   ✅ Excellent model performance achieved`);
        }
        else if (avgAccuracy > 0.6) {
            logger_1.logger.info(`   ✅ Good model performance achieved`);
        }
        else {
            logger_1.logger.info(`   ⚠️ Model performance could be improved`);
        }
        logger_1.logger.info('\n🚀 NEXT STEPS:');
        logger_1.logger.info('   1. Run comprehensive multi-timeframe backtest with updated models');
        logger_1.logger.info('   2. Compare performance against previous model versions');
        logger_1.logger.info('   3. Validate models on out-of-sample data');
        logger_1.logger.info('   4. Consider ensemble methods for improved predictions');
        logger_1.logger.info('   5. Monitor model performance in live trading environment');
        logger_1.logger.info('='.repeat(120));
    }
    /**
     * Validate model improvements
     */
    async validateModelImprovements(trainedModels) {
        logger_1.logger.info('\n🔍 Validating model improvements...');
        // Check if models meet minimum performance thresholds
        const minAccuracy = 0.6; // 60% minimum accuracy
        const minF1Score = 0.55; // 55% minimum F1 score
        let passedModels = 0;
        let totalModels = Object.keys(trainedModels).length;
        Object.values(trainedModels).forEach(model => {
            const meetsAccuracy = model.finalMetrics.testAccuracy >= minAccuracy;
            const meetsF1 = model.finalMetrics.f1Score >= minF1Score;
            if (meetsAccuracy && meetsF1) {
                passedModels++;
                logger_1.logger.info(`   ✅ ${model.modelName}: Meets performance thresholds`);
            }
            else {
                logger_1.logger.warn(`   ⚠️ ${model.modelName}: Below performance thresholds`, {
                    accuracy: `${(model.finalMetrics.testAccuracy * 100).toFixed(1)}%`,
                    f1Score: model.finalMetrics.f1Score.toFixed(3),
                });
            }
        });
        const passRate = (passedModels / totalModels) * 100;
        logger_1.logger.info(`\n📊 Model Validation Results: ${passedModels}/${totalModels} models passed (${passRate.toFixed(1)}%)`);
        if (passRate >= 80) {
            logger_1.logger.info('🎉 Excellent model training results - ready for deployment');
        }
        else if (passRate >= 60) {
            logger_1.logger.info('✅ Good model training results - consider additional optimization');
        }
        else {
            logger_1.logger.warn('⚠️ Model training results below expectations - review training process');
        }
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        logger_1.logger.info('🧹 AI model retraining cleanup completed');
    }
}
exports.AIModelRetrainingRunner = AIModelRetrainingRunner;
/**
 * Main execution function
 */
async function main() {
    const runner = new AIModelRetrainingRunner();
    try {
        await runner.runRetraining();
    }
    catch (error) {
        logger_1.logger.error('💥 AI model retraining failed:', error);
        process.exit(1);
    }
    finally {
        await runner.cleanup();
    }
}
// Handle graceful shutdown
process.on('SIGINT', async () => {
    logger_1.logger.info('🛑 Received SIGINT, cleaning up...');
    const runner = new AIModelRetrainingRunner();
    await runner.cleanup();
    process.exit(0);
});
process.on('SIGTERM', async () => {
    logger_1.logger.info('🛑 Received SIGTERM, cleaning up...');
    const runner = new AIModelRetrainingRunner();
    await runner.cleanup();
    process.exit(0);
});
// Run if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=retrain-ai-models.js.map