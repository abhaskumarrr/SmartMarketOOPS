/*
  Warnings:

  - You are about to drop the column `device` on the `Session` table. All the data in the column will be lost.
  - You are about to drop the column `lastActive` on the `Session` table. All the data in the column will be lost.
  - A unique constraint covering the columns `[token]` on the table `Session` will be added. If there are existing duplicate values, this will fail.
  - A unique constraint covering the columns `[refreshToken]` on the table `Session` will be added. If there are existing duplicate values, this will fail.
  - Added the required column `token` to the `Session` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "Session" DROP COLUMN "device",
DROP COLUMN "lastActive",
ADD COLUMN     "lastActiveAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN     "metadata" JSONB,
ADD COLUMN     "refreshToken" TEXT,
ADD COLUMN     "rememberMe" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "token" TEXT NOT NULL,
ADD COLUMN     "userAgent" TEXT;

-- CreateTable
CREATE TABLE "TradingSignal" (
    "id" TEXT NOT NULL,
    "symbol" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "direction" TEXT NOT NULL,
    "strength" TEXT NOT NULL,
    "timeframe" TEXT NOT NULL,
    "price" DOUBLE PRECISION NOT NULL,
    "targetPrice" DOUBLE PRECISION,
    "stopLoss" DOUBLE PRECISION,
    "confidenceScore" INTEGER NOT NULL,
    "expectedReturn" DOUBLE PRECISION NOT NULL,
    "expectedRisk" DOUBLE PRECISION NOT NULL,
    "riskRewardRatio" DOUBLE PRECISION NOT NULL,
    "generatedAt" TIMESTAMP(3) NOT NULL,
    "expiresAt" TIMESTAMP(3),
    "source" TEXT NOT NULL,
    "metadata" JSONB NOT NULL,
    "predictionValues" JSONB NOT NULL,
    "validatedAt" TIMESTAMP(3),
    "validationStatus" BOOLEAN NOT NULL DEFAULT false,
    "validationReason" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "TradingSignal_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "TradingSignal_symbol_idx" ON "TradingSignal"("symbol");

-- CreateIndex
CREATE INDEX "TradingSignal_type_idx" ON "TradingSignal"("type");

-- CreateIndex
CREATE INDEX "TradingSignal_direction_idx" ON "TradingSignal"("direction");

-- CreateIndex
CREATE INDEX "TradingSignal_strength_idx" ON "TradingSignal"("strength");

-- CreateIndex
CREATE INDEX "TradingSignal_timeframe_idx" ON "TradingSignal"("timeframe");

-- CreateIndex
CREATE INDEX "TradingSignal_generatedAt_idx" ON "TradingSignal"("generatedAt");

-- CreateIndex
CREATE INDEX "TradingSignal_expiresAt_idx" ON "TradingSignal"("expiresAt");

-- CreateIndex
CREATE INDEX "TradingSignal_confidenceScore_idx" ON "TradingSignal"("confidenceScore");

-- CreateIndex
CREATE INDEX "TradingSignal_validationStatus_idx" ON "TradingSignal"("validationStatus");

-- CreateIndex
CREATE UNIQUE INDEX "Session_token_key" ON "Session"("token");

-- CreateIndex
CREATE UNIQUE INDEX "Session_refreshToken_key" ON "Session"("refreshToken");

-- CreateIndex
CREATE INDEX "Session_token_idx" ON "Session"("token");
