-- CreateTable
CREATE TABLE "Order" (
    "id" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "symbol" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "side" TEXT NOT NULL,
    "quantity" DOUBLE PRECISION NOT NULL,
    "price" DOUBLE PRECISION,
    "stopPrice" DOUBLE PRECISION,
    "avgFillPrice" DOUBLE PRECISION,
    "filledQuantity" DOUBLE PRECISION NOT NULL,
    "remainingQuantity" DOUBLE PRECISION NOT NULL,
    "fee" DOUBLE PRECISION,
    "feeCurrency" TEXT,
    "clientOrderId" TEXT,
    "exchangeOrderId" TEXT,
    "source" TEXT NOT NULL,
    "exchangeId" TEXT NOT NULL,
    "submittedAt" TIMESTAMP(3) NOT NULL,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "completedAt" TIMESTAMP(3),
    "errorCode" TEXT,
    "errorMessage" TEXT,
    "errorDetails" JSONB,
    "raw" JSONB,
    "userId" TEXT NOT NULL,
    "positionId" TEXT,
    "strategyId" TEXT,
    "botId" TEXT,
    "signalId" TEXT,

    CONSTRAINT "Order_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Order_userId_idx" ON "Order"("userId");

-- CreateIndex
CREATE INDEX "Order_positionId_idx" ON "Order"("positionId");

-- CreateIndex
CREATE INDEX "Order_strategyId_idx" ON "Order"("strategyId");

-- CreateIndex
CREATE INDEX "Order_botId_idx" ON "Order"("botId");

-- CreateIndex
CREATE INDEX "Order_signalId_idx" ON "Order"("signalId");

-- CreateIndex
CREATE INDEX "Order_symbol_idx" ON "Order"("symbol");

-- CreateIndex
CREATE INDEX "Order_status_idx" ON "Order"("status");

-- CreateIndex
CREATE INDEX "Order_submittedAt_idx" ON "Order"("submittedAt");

-- AddForeignKey
ALTER TABLE "Order" ADD CONSTRAINT "Order_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Order" ADD CONSTRAINT "Order_positionId_fkey" FOREIGN KEY ("positionId") REFERENCES "Position"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Order" ADD CONSTRAINT "Order_strategyId_fkey" FOREIGN KEY ("strategyId") REFERENCES "TradingStrategy"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Order" ADD CONSTRAINT "Order_botId_fkey" FOREIGN KEY ("botId") REFERENCES "Bot"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Order" ADD CONSTRAINT "Order_signalId_fkey" FOREIGN KEY ("signalId") REFERENCES "TradingSignal"("id") ON DELETE SET NULL ON UPDATE CASCADE;
