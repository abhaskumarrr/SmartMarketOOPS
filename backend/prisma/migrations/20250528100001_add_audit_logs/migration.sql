-- CreateTable
CREATE TABLE "DecisionLog" (
  "id" TEXT NOT NULL,
  "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "source" TEXT NOT NULL,
  "actionType" TEXT NOT NULL,
  "decision" TEXT NOT NULL,
  "reasonCode" TEXT,
  "reasonDetails" TEXT,
  "confidence" DOUBLE PRECISION,
  "dataSnapshot" JSONB,
  "parameters" JSONB,
  "modelVersion" TEXT,
  "userId" TEXT,
  "strategyId" TEXT,
  "botId" TEXT,
  "signalId" TEXT,
  "orderId" TEXT,
  "symbol" TEXT,
  "outcome" TEXT,
  "outcomeDetails" JSONB,
  "pnl" DOUBLE PRECISION,
  "evaluatedAt" TIMESTAMP(3),
  "tags" TEXT[],
  "importance" TEXT NOT NULL DEFAULT 'NORMAL',
  "notes" TEXT,
  "auditTrailId" TEXT,

  PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AuditTrail" (
  "id" TEXT NOT NULL,
  "trailType" TEXT NOT NULL,
  "entityId" TEXT NOT NULL,
  "entityType" TEXT NOT NULL,
  "startTime" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "endTime" TIMESTAMP(3),
  "status" TEXT NOT NULL DEFAULT 'ACTIVE',
  "summary" TEXT,
  "userId" TEXT,
  "orderId" TEXT,
  "tags" TEXT[],
  "metadata" JSONB,

  PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AuditEvent" (
  "id" TEXT NOT NULL,
  "auditTrailId" TEXT NOT NULL,
  "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "eventType" TEXT NOT NULL,
  "component" TEXT NOT NULL,
  "action" TEXT NOT NULL,
  "status" TEXT NOT NULL,
  "details" JSONB,
  "dataBefore" JSONB,
  "dataAfter" JSONB,
  "metadata" JSONB,

  PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "DecisionLog_timestamp_idx" ON "DecisionLog"("timestamp");

-- CreateIndex
CREATE INDEX "DecisionLog_source_idx" ON "DecisionLog"("source");

-- CreateIndex
CREATE INDEX "DecisionLog_actionType_idx" ON "DecisionLog"("actionType");

-- CreateIndex
CREATE INDEX "DecisionLog_userId_idx" ON "DecisionLog"("userId");

-- CreateIndex
CREATE INDEX "DecisionLog_strategyId_idx" ON "DecisionLog"("strategyId");

-- CreateIndex
CREATE INDEX "DecisionLog_botId_idx" ON "DecisionLog"("botId");

-- CreateIndex
CREATE INDEX "DecisionLog_signalId_idx" ON "DecisionLog"("signalId");

-- CreateIndex
CREATE INDEX "DecisionLog_orderId_idx" ON "DecisionLog"("orderId");

-- CreateIndex
CREATE INDEX "DecisionLog_symbol_idx" ON "DecisionLog"("symbol");

-- CreateIndex
CREATE INDEX "DecisionLog_outcome_idx" ON "DecisionLog"("outcome");

-- CreateIndex
CREATE INDEX "DecisionLog_importance_idx" ON "DecisionLog"("importance");

-- CreateIndex
CREATE INDEX "DecisionLog_tags_idx" ON "DecisionLog"("tags");

-- CreateIndex
CREATE INDEX "AuditTrail_trailType_idx" ON "AuditTrail"("trailType");

-- CreateIndex
CREATE INDEX "AuditTrail_entityId_idx" ON "AuditTrail"("entityId");

-- CreateIndex
CREATE INDEX "AuditTrail_entityType_idx" ON "AuditTrail"("entityType");

-- CreateIndex
CREATE INDEX "AuditTrail_startTime_idx" ON "AuditTrail"("startTime");

-- CreateIndex
CREATE INDEX "AuditTrail_status_idx" ON "AuditTrail"("status");

-- CreateIndex
CREATE INDEX "AuditTrail_userId_idx" ON "AuditTrail"("userId");

-- CreateIndex
CREATE INDEX "AuditTrail_orderId_idx" ON "AuditTrail"("orderId");

-- CreateIndex
CREATE INDEX "AuditTrail_tags_idx" ON "AuditTrail"("tags");

-- CreateIndex
CREATE INDEX "AuditEvent_auditTrailId_idx" ON "AuditEvent"("auditTrailId");

-- CreateIndex
CREATE INDEX "AuditEvent_timestamp_idx" ON "AuditEvent"("timestamp");

-- CreateIndex
CREATE INDEX "AuditEvent_eventType_idx" ON "AuditEvent"("eventType");

-- CreateIndex
CREATE INDEX "AuditEvent_component_idx" ON "AuditEvent"("component");

-- CreateIndex
CREATE INDEX "AuditEvent_status_idx" ON "AuditEvent"("status");

-- AddForeignKey
ALTER TABLE "DecisionLog" ADD CONSTRAINT "DecisionLog_auditTrailId_fkey" FOREIGN KEY ("auditTrailId") REFERENCES "AuditTrail"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AuditTrail" ADD CONSTRAINT "AuditTrail_orderId_fkey" FOREIGN KEY ("orderId") REFERENCES "Order"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AuditEvent" ADD CONSTRAINT "AuditEvent_auditTrailId_fkey" FOREIGN KEY ("auditTrailId") REFERENCES "AuditTrail"("id") ON DELETE CASCADE ON UPDATE CASCADE; 