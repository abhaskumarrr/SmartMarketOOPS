-- DropIndex
DROP INDEX "AuditTrail_orderId_idx";

-- AlterTable
ALTER TABLE "ApiKey" ADD COLUMN     "environment" TEXT NOT NULL DEFAULT 'testnet',
ADD COLUMN     "hashedSecret" TEXT,
ADD COLUMN     "ipRestrictions" TEXT,
ADD COLUMN     "lastUsedAt" TIMESTAMP(3),
ADD COLUMN     "metadata" JSONB,
ADD COLUMN     "name" TEXT NOT NULL DEFAULT 'Default',
ADD COLUMN     "rateLimits" JSONB,
ADD COLUMN     "revokedAt" TIMESTAMP(3),
ADD COLUMN     "revokedBy" TEXT,
ADD COLUMN     "revokedReason" TEXT,
ADD COLUMN     "usageCount" INTEGER NOT NULL DEFAULT 0;

-- CreateIndex
CREATE INDEX "ApiKey_environment_idx" ON "ApiKey"("environment");

-- CreateIndex
CREATE INDEX "ApiKey_isRevoked_idx" ON "ApiKey"("isRevoked");

-- CreateIndex
CREATE INDEX "ApiKey_expiry_idx" ON "ApiKey"("expiry");

-- CreateIndex
CREATE INDEX "ApiKey_createdAt_idx" ON "ApiKey"("createdAt");

-- CreateIndex
CREATE INDEX "ApiKey_lastUsedAt_idx" ON "ApiKey"("lastUsedAt");
