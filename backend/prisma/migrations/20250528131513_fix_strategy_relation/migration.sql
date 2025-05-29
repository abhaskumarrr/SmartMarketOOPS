/*
  Warnings:

  - You are about to drop the `DecisionLog` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "DecisionLog" DROP CONSTRAINT "DecisionLog_auditTrailId_fkey";

-- DropTable
DROP TABLE "DecisionLog";

-- CreateTable
CREATE TABLE "decision_logs" (
    "id" TEXT NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "source" TEXT NOT NULL,
    "actionType" TEXT NOT NULL,
    "decision" TEXT NOT NULL,
    "reasonDetails" TEXT,
    "userId" TEXT,
    "botId" TEXT,
    "strategyId" TEXT,
    "symbol" TEXT,
    "orderId" TEXT,
    "positionId" TEXT,
    "importance" TEXT NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "tags" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "auditTrailId" TEXT,

    CONSTRAINT "decision_logs_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "decision_logs_timestamp_idx" ON "decision_logs"("timestamp");

-- CreateIndex
CREATE INDEX "decision_logs_userId_idx" ON "decision_logs"("userId");

-- CreateIndex
CREATE INDEX "decision_logs_botId_idx" ON "decision_logs"("botId");

-- CreateIndex
CREATE INDEX "decision_logs_strategyId_idx" ON "decision_logs"("strategyId");

-- CreateIndex
CREATE INDEX "decision_logs_symbol_idx" ON "decision_logs"("symbol");

-- AddForeignKey
ALTER TABLE "decision_logs" ADD CONSTRAINT "decision_logs_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "decision_logs" ADD CONSTRAINT "decision_logs_botId_fkey" FOREIGN KEY ("botId") REFERENCES "Bot"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "decision_logs" ADD CONSTRAINT "decision_logs_strategyId_fkey" FOREIGN KEY ("strategyId") REFERENCES "TradingStrategy"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "decision_logs" ADD CONSTRAINT "decision_logs_auditTrailId_fkey" FOREIGN KEY ("auditTrailId") REFERENCES "AuditTrail"("id") ON DELETE SET NULL ON UPDATE CASCADE;
