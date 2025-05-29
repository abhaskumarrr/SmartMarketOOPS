-- CreateTable
CREATE TABLE "RiskSettings" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "userId" TEXT NOT NULL,
    "botId" TEXT,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "positionSizingMethod" TEXT NOT NULL,
    "riskPercentage" DOUBLE PRECISION NOT NULL,
    "maxPositionSize" DOUBLE PRECISION NOT NULL,
    "kellyFraction" DOUBLE PRECISION,
    "winRate" DOUBLE PRECISION,
    "customSizingParams" JSONB,
    "stopLossType" TEXT NOT NULL,
    "stopLossValue" DOUBLE PRECISION NOT NULL,
    "trailingCallback" DOUBLE PRECISION,
    "trailingStep" DOUBLE PRECISION,
    "timeLimit" INTEGER,
    "stopLossLevels" JSONB,
    "takeProfitType" TEXT NOT NULL,
    "takeProfitValue" DOUBLE PRECISION NOT NULL,
    "trailingActivation" DOUBLE PRECISION,
    "takeProfitLevels" JSONB,
    "maxRiskPerTrade" DOUBLE PRECISION NOT NULL,
    "maxRiskPerSymbol" DOUBLE PRECISION NOT NULL,
    "maxRiskPerDirection" DOUBLE PRECISION NOT NULL,
    "maxTotalRisk" DOUBLE PRECISION NOT NULL,
    "maxDrawdown" DOUBLE PRECISION NOT NULL,
    "maxPositions" INTEGER NOT NULL,
    "maxDailyLoss" DOUBLE PRECISION NOT NULL,
    "cooldownPeriod" INTEGER NOT NULL,
    "volatilityLookback" INTEGER NOT NULL,
    "circuitBreakerEnabled" BOOLEAN NOT NULL DEFAULT true,
    "maxDailyLossBreaker" DOUBLE PRECISION NOT NULL,
    "maxDrawdownBreaker" DOUBLE PRECISION NOT NULL,
    "volatilityMultiplier" DOUBLE PRECISION NOT NULL,
    "consecutiveLossesBreaker" INTEGER NOT NULL,
    "tradingPause" INTEGER NOT NULL,
    "marketWideEnabled" BOOLEAN NOT NULL DEFAULT true,
    "enableManualOverride" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "RiskSettings_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "RiskAlert" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "level" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "details" JSONB NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "acknowledged" BOOLEAN NOT NULL DEFAULT false,
    "resolvedAt" TIMESTAMP(3),

    CONSTRAINT "RiskAlert_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "RiskSettings_userId_idx" ON "RiskSettings"("userId");

-- CreateIndex
CREATE INDEX "RiskSettings_botId_idx" ON "RiskSettings"("botId");

-- CreateIndex
CREATE INDEX "RiskSettings_isActive_idx" ON "RiskSettings"("isActive");

-- CreateIndex
CREATE INDEX "RiskAlert_userId_idx" ON "RiskAlert"("userId");

-- CreateIndex
CREATE INDEX "RiskAlert_type_idx" ON "RiskAlert"("type");

-- CreateIndex
CREATE INDEX "RiskAlert_level_idx" ON "RiskAlert"("level");

-- CreateIndex
CREATE INDEX "RiskAlert_acknowledged_idx" ON "RiskAlert"("acknowledged");

-- CreateIndex
CREATE INDEX "RiskAlert_timestamp_idx" ON "RiskAlert"("timestamp");

-- AddForeignKey
ALTER TABLE "RiskSettings" ADD CONSTRAINT "RiskSettings_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "RiskSettings" ADD CONSTRAINT "RiskSettings_botId_fkey" FOREIGN KEY ("botId") REFERENCES "Bot"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "RiskAlert" ADD CONSTRAINT "RiskAlert_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
