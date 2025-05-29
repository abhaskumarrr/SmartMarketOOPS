-- CreateTable
CREATE TABLE "CircuitBreaker" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "userId" TEXT NOT NULL,
    "botId" TEXT,
    "type" TEXT NOT NULL,
    "isGlobal" BOOLEAN NOT NULL DEFAULT false,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "threshold" DOUBLE PRECISION NOT NULL,
    "recoveryThreshold" DOUBLE PRECISION,
    "cooldownMinutes" INTEGER NOT NULL DEFAULT 60,
    "action" TEXT NOT NULL,
    "lastTriggered" TIMESTAMP(3),
    "status" TEXT NOT NULL DEFAULT 'READY',
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "CircuitBreaker_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "TradingStrategy" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "userId" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "timeHorizon" TEXT NOT NULL,
    "symbols" TEXT[],
    "entryRules" JSONB NOT NULL,
    "exitRules" JSONB NOT NULL,
    "positionSizing" JSONB NOT NULL,
    "riskManagement" JSONB NOT NULL,
    "indicators" JSONB NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT false,
    "isPublic" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "TradingStrategy_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "StrategyExecution" (
    "id" TEXT NOT NULL,
    "strategyId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "botId" TEXT,
    "status" TEXT NOT NULL,
    "lastExecutedAt" TIMESTAMP(3),
    "startedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "stoppedAt" TIMESTAMP(3),
    "currentPositions" TEXT[],
    "historicalPositions" TEXT[],
    "performance" JSONB NOT NULL,
    "logs" JSONB NOT NULL,
    "errors" JSONB NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "StrategyExecution_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "StrategyExecutionResult" (
    "id" TEXT NOT NULL,
    "executionId" TEXT NOT NULL,
    "signalId" TEXT,
    "action" TEXT NOT NULL,
    "entryRuleResults" JSONB NOT NULL,
    "exitRuleResults" JSONB NOT NULL,
    "positionSize" DOUBLE PRECISION,
    "entryPrice" DOUBLE PRECISION,
    "stopLossPrice" DOUBLE PRECISION,
    "takeProfitPrice" DOUBLE PRECISION,
    "confidence" INTEGER NOT NULL,
    "notes" TEXT,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "StrategyExecutionResult_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "CircuitBreaker_userId_idx" ON "CircuitBreaker"("userId");

-- CreateIndex
CREATE INDEX "CircuitBreaker_botId_idx" ON "CircuitBreaker"("botId");

-- CreateIndex
CREATE INDEX "CircuitBreaker_isActive_idx" ON "CircuitBreaker"("isActive");

-- CreateIndex
CREATE INDEX "CircuitBreaker_type_idx" ON "CircuitBreaker"("type");

-- CreateIndex
CREATE INDEX "CircuitBreaker_status_idx" ON "CircuitBreaker"("status");

-- CreateIndex
CREATE INDEX "TradingStrategy_userId_idx" ON "TradingStrategy"("userId");

-- CreateIndex
CREATE INDEX "TradingStrategy_isActive_idx" ON "TradingStrategy"("isActive");

-- CreateIndex
CREATE INDEX "TradingStrategy_isPublic_idx" ON "TradingStrategy"("isPublic");

-- CreateIndex
CREATE INDEX "TradingStrategy_type_idx" ON "TradingStrategy"("type");

-- CreateIndex
CREATE INDEX "TradingStrategy_timeHorizon_idx" ON "TradingStrategy"("timeHorizon");

-- CreateIndex
CREATE INDEX "StrategyExecution_strategyId_idx" ON "StrategyExecution"("strategyId");

-- CreateIndex
CREATE INDEX "StrategyExecution_userId_idx" ON "StrategyExecution"("userId");

-- CreateIndex
CREATE INDEX "StrategyExecution_botId_idx" ON "StrategyExecution"("botId");

-- CreateIndex
CREATE INDEX "StrategyExecution_status_idx" ON "StrategyExecution"("status");

-- CreateIndex
CREATE INDEX "StrategyExecution_startedAt_idx" ON "StrategyExecution"("startedAt");

-- CreateIndex
CREATE INDEX "StrategyExecutionResult_executionId_idx" ON "StrategyExecutionResult"("executionId");

-- CreateIndex
CREATE INDEX "StrategyExecutionResult_signalId_idx" ON "StrategyExecutionResult"("signalId");

-- CreateIndex
CREATE INDEX "StrategyExecutionResult_action_idx" ON "StrategyExecutionResult"("action");

-- CreateIndex
CREATE INDEX "StrategyExecutionResult_timestamp_idx" ON "StrategyExecutionResult"("timestamp");

-- AddForeignKey
ALTER TABLE "StrategyExecution" ADD CONSTRAINT "StrategyExecution_strategyId_fkey" FOREIGN KEY ("strategyId") REFERENCES "TradingStrategy"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StrategyExecutionResult" ADD CONSTRAINT "StrategyExecutionResult_executionId_fkey" FOREIGN KEY ("executionId") REFERENCES "StrategyExecution"("id") ON DELETE CASCADE ON UPDATE CASCADE;
