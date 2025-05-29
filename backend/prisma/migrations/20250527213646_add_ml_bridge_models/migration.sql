/*
  Warnings:

  - You are about to drop the column `botId` on the `StrategyExecution` table. All the data in the column will be lost.
  - You are about to drop the column `currentPositions` on the `StrategyExecution` table. All the data in the column will be lost.
  - You are about to drop the column `errors` on the `StrategyExecution` table. All the data in the column will be lost.
  - You are about to drop the column `historicalPositions` on the `StrategyExecution` table. All the data in the column will be lost.
  - You are about to drop the column `lastExecutedAt` on the `StrategyExecution` table. All the data in the column will be lost.
  - You are about to drop the column `logs` on the `StrategyExecution` table. All the data in the column will be lost.
  - You are about to drop the column `performance` on the `StrategyExecution` table. All the data in the column will be lost.
  - You are about to drop the column `action` on the `StrategyExecutionResult` table. All the data in the column will be lost.
  - You are about to drop the column `confidence` on the `StrategyExecutionResult` table. All the data in the column will be lost.
  - You are about to drop the column `entryRuleResults` on the `StrategyExecutionResult` table. All the data in the column will be lost.
  - You are about to drop the column `exitRuleResults` on the `StrategyExecutionResult` table. All the data in the column will be lost.
  - You are about to drop the column `notes` on the `StrategyExecutionResult` table. All the data in the column will be lost.
  - You are about to drop the column `positionSize` on the `StrategyExecutionResult` table. All the data in the column will be lost.
  - You are about to drop the column `signalId` on the `StrategyExecutionResult` table. All the data in the column will be lost.
  - You are about to drop the column `stopLossPrice` on the `StrategyExecutionResult` table. All the data in the column will be lost.
  - You are about to drop the column `takeProfitPrice` on the `StrategyExecutionResult` table. All the data in the column will be lost.
  - You are about to drop the column `timestamp` on the `StrategyExecutionResult` table. All the data in the column will be lost.
  - You are about to drop the column `indicators` on the `TradingStrategy` table. All the data in the column will be lost.
  - Added the required column `direction` to the `StrategyExecutionResult` table without a default value. This is not possible if the table is not empty.
  - Added the required column `status` to the `StrategyExecutionResult` table without a default value. This is not possible if the table is not empty.
  - Added the required column `symbol` to the `StrategyExecutionResult` table without a default value. This is not possible if the table is not empty.
  - Added the required column `updatedAt` to the `StrategyExecutionResult` table without a default value. This is not possible if the table is not empty.

*/
-- DropForeignKey
ALTER TABLE "StrategyExecution" DROP CONSTRAINT "StrategyExecution_strategyId_fkey";

-- DropForeignKey
ALTER TABLE "StrategyExecutionResult" DROP CONSTRAINT "StrategyExecutionResult_executionId_fkey";

-- DropIndex
DROP INDEX "StrategyExecution_botId_idx";

-- DropIndex
DROP INDEX "StrategyExecution_startedAt_idx";

-- DropIndex
DROP INDEX "StrategyExecutionResult_action_idx";

-- DropIndex
DROP INDEX "StrategyExecutionResult_signalId_idx";

-- DropIndex
DROP INDEX "StrategyExecutionResult_timestamp_idx";

-- DropIndex
DROP INDEX "TradingStrategy_isPublic_idx";

-- DropIndex
DROP INDEX "TradingStrategy_timeHorizon_idx";

-- DropIndex
DROP INDEX "TradingStrategy_type_idx";

-- AlterTable
ALTER TABLE "StrategyExecution" DROP COLUMN "botId",
DROP COLUMN "currentPositions",
DROP COLUMN "errors",
DROP COLUMN "historicalPositions",
DROP COLUMN "lastExecutedAt",
DROP COLUMN "logs",
DROP COLUMN "performance",
ADD COLUMN     "metadata" JSONB,
ADD COLUMN     "runningTimeMs" INTEGER;

-- AlterTable
ALTER TABLE "StrategyExecutionResult" DROP COLUMN "action",
DROP COLUMN "confidence",
DROP COLUMN "entryRuleResults",
DROP COLUMN "exitRuleResults",
DROP COLUMN "notes",
DROP COLUMN "positionSize",
DROP COLUMN "signalId",
DROP COLUMN "stopLossPrice",
DROP COLUMN "takeProfitPrice",
DROP COLUMN "timestamp",
ADD COLUMN     "direction" TEXT NOT NULL,
ADD COLUMN     "entryTime" TIMESTAMP(3),
ADD COLUMN     "exitPrice" DOUBLE PRECISION,
ADD COLUMN     "exitTime" TIMESTAMP(3),
ADD COLUMN     "metadata" JSONB,
ADD COLUMN     "profitLoss" DOUBLE PRECISION,
ADD COLUMN     "profitLossPercentage" DOUBLE PRECISION,
ADD COLUMN     "quantity" DOUBLE PRECISION,
ADD COLUMN     "status" TEXT NOT NULL,
ADD COLUMN     "symbol" TEXT NOT NULL,
ADD COLUMN     "updatedAt" TIMESTAMP(3) NOT NULL;

-- AlterTable
ALTER TABLE "TradingStrategy" DROP COLUMN "indicators",
ADD COLUMN     "metadata" JSONB,
ALTER COLUMN "isActive" SET DEFAULT true;

-- CreateTable
CREATE TABLE "MLModel" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "version" TEXT NOT NULL,
    "modelType" TEXT NOT NULL,
    "symbol" TEXT,
    "timeframe" TEXT,
    "description" TEXT,
    "status" TEXT NOT NULL DEFAULT 'ACTIVE',
    "accuracy" DOUBLE PRECISION,
    "precision" DOUBLE PRECISION,
    "recall" DOUBLE PRECISION,
    "f1Score" DOUBLE PRECISION,
    "trainedAt" TIMESTAMP(3),
    "lastUsedAt" TIMESTAMP(3),
    "trainingId" TEXT,
    "location" TEXT,
    "params" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MLModel_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MLPrediction" (
    "id" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "symbol" TEXT NOT NULL,
    "timeframe" TEXT NOT NULL,
    "predictionType" TEXT NOT NULL,
    "values" DOUBLE PRECISION[],
    "timestamps" TEXT[],
    "confidenceScores" DOUBLE PRECISION[],
    "metadata" JSONB,
    "generatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMP(3),
    "signalGenerated" BOOLEAN NOT NULL DEFAULT false,
    "signalId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MLPrediction_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MLTrainingJob" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "symbol" TEXT NOT NULL,
    "timeframe" TEXT NOT NULL,
    "modelType" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'QUEUED',
    "progress" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "startedAt" TIMESTAMP(3),
    "completedAt" TIMESTAMP(3),
    "resultModelId" TEXT,
    "errorMessage" TEXT,
    "params" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MLTrainingJob_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "BridgeConfig" (
    "id" TEXT NOT NULL,
    "userId" TEXT,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "mlModelId" TEXT,
    "autoGenerateSignals" BOOLEAN NOT NULL DEFAULT false,
    "confidenceThreshold" DOUBLE PRECISION NOT NULL DEFAULT 70,
    "signalExpiryMinutes" INTEGER NOT NULL DEFAULT 1440,
    "refreshIntervalMinutes" INTEGER NOT NULL DEFAULT 60,
    "symbols" TEXT[],
    "timeframes" TEXT[],
    "lastExecutedAt" TIMESTAMP(3),
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "BridgeConfig_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "MLModel_symbol_timeframe_idx" ON "MLModel"("symbol", "timeframe");

-- CreateIndex
CREATE UNIQUE INDEX "MLModel_name_version_key" ON "MLModel"("name", "version");

-- CreateIndex
CREATE INDEX "MLPrediction_symbol_timeframe_idx" ON "MLPrediction"("symbol", "timeframe");

-- CreateIndex
CREATE INDEX "MLPrediction_modelId_idx" ON "MLPrediction"("modelId");

-- CreateIndex
CREATE INDEX "MLPrediction_signalId_idx" ON "MLPrediction"("signalId");

-- CreateIndex
CREATE INDEX "MLTrainingJob_userId_idx" ON "MLTrainingJob"("userId");

-- CreateIndex
CREATE INDEX "MLTrainingJob_status_idx" ON "MLTrainingJob"("status");

-- CreateIndex
CREATE INDEX "MLTrainingJob_symbol_timeframe_idx" ON "MLTrainingJob"("symbol", "timeframe");

-- CreateIndex
CREATE INDEX "BridgeConfig_isActive_idx" ON "BridgeConfig"("isActive");

-- CreateIndex
CREATE UNIQUE INDEX "BridgeConfig_userId_name_key" ON "BridgeConfig"("userId", "name");

-- CreateIndex
CREATE INDEX "StrategyExecutionResult_symbol_idx" ON "StrategyExecutionResult"("symbol");

-- CreateIndex
CREATE INDEX "StrategyExecutionResult_status_idx" ON "StrategyExecutionResult"("status");

-- AddForeignKey
ALTER TABLE "StrategyExecution" ADD CONSTRAINT "StrategyExecution_strategyId_fkey" FOREIGN KEY ("strategyId") REFERENCES "TradingStrategy"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StrategyExecutionResult" ADD CONSTRAINT "StrategyExecutionResult_executionId_fkey" FOREIGN KEY ("executionId") REFERENCES "StrategyExecution"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MLPrediction" ADD CONSTRAINT "MLPrediction_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "MLModel"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
