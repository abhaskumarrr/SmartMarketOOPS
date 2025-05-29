/*
  Warnings:

  - Added the required column `status` to the `TradeLog` table without a default value. This is not possible if the table is not empty.
  - Added the required column `type` to the `TradeLog` table without a default value. This is not possible if the table is not empty.

*/
-- DropForeignKey
ALTER TABLE "ApiKey" DROP CONSTRAINT "ApiKey_userId_fkey";

-- DropForeignKey
ALTER TABLE "Bot" DROP CONSTRAINT "Bot_userId_fkey";

-- DropForeignKey
ALTER TABLE "TradeLog" DROP CONSTRAINT "TradeLog_userId_fkey";

-- AlterTable
ALTER TABLE "ApiKey" ADD COLUMN     "isRevoked" BOOLEAN NOT NULL DEFAULT false;

-- AlterTable
ALTER TABLE "Metric" ADD COLUMN     "tags" JSONB;

-- AlterTable
ALTER TABLE "TradeLog" ADD COLUMN     "orderId" TEXT,
ADD COLUMN     "status" TEXT NOT NULL,
ADD COLUMN     "type" TEXT NOT NULL;

-- CreateTable
CREATE TABLE "Position" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "botId" TEXT,
    "symbol" TEXT NOT NULL,
    "side" TEXT NOT NULL,
    "entryPrice" DOUBLE PRECISION NOT NULL,
    "currentPrice" DOUBLE PRECISION,
    "amount" DOUBLE PRECISION NOT NULL,
    "leverage" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    "takeProfitPrice" DOUBLE PRECISION,
    "stopLossPrice" DOUBLE PRECISION,
    "status" TEXT NOT NULL,
    "pnl" DOUBLE PRECISION,
    "openedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "closedAt" TIMESTAMP(3),
    "metadata" JSONB,

    CONSTRAINT "Position_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Position_userId_idx" ON "Position"("userId");

-- CreateIndex
CREATE INDEX "Position_botId_idx" ON "Position"("botId");

-- CreateIndex
CREATE INDEX "Position_symbol_idx" ON "Position"("symbol");

-- CreateIndex
CREATE INDEX "Position_status_idx" ON "Position"("status");

-- CreateIndex
CREATE INDEX "ApiKey_userId_idx" ON "ApiKey"("userId");

-- CreateIndex
CREATE INDEX "ApiKey_key_idx" ON "ApiKey"("key");

-- CreateIndex
CREATE INDEX "Bot_userId_idx" ON "Bot"("userId");

-- CreateIndex
CREATE INDEX "Bot_symbol_idx" ON "Bot"("symbol");

-- CreateIndex
CREATE INDEX "Bot_isActive_idx" ON "Bot"("isActive");

-- CreateIndex
CREATE INDEX "Metric_name_recordedAt_idx" ON "Metric"("name", "recordedAt" DESC);

-- CreateIndex
CREATE INDEX "TradeLog_userId_timestamp_idx" ON "TradeLog"("userId", "timestamp" DESC);

-- CreateIndex
CREATE INDEX "TradeLog_instrument_timestamp_idx" ON "TradeLog"("instrument", "timestamp" DESC);

-- CreateIndex
CREATE INDEX "User_email_idx" ON "User"("email");

-- AddForeignKey
ALTER TABLE "ApiKey" ADD CONSTRAINT "ApiKey_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TradeLog" ADD CONSTRAINT "TradeLog_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Bot" ADD CONSTRAINT "Bot_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Position" ADD CONSTRAINT "Position_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Position" ADD CONSTRAINT "Position_botId_fkey" FOREIGN KEY ("botId") REFERENCES "Bot"("id") ON DELETE SET NULL ON UPDATE CASCADE;
