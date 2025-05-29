-- CreateTable
CREATE TABLE "PerformanceTest" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "testType" TEXT NOT NULL,
    "duration" INTEGER NOT NULL,
    "concurrency" INTEGER NOT NULL,
    "rampUp" INTEGER,
    "targetEndpoint" TEXT,
    "modelId" TEXT,
    "strategyId" TEXT,
    "symbol" TEXT,
    "timeframe" TEXT,
    "options" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "PerformanceTest_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PerformanceTestResult" (
    "id" TEXT NOT NULL,
    "testId" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "startTime" TIMESTAMP(3) NOT NULL,
    "endTime" TIMESTAMP(3),
    "duration" INTEGER,
    "metrics" JSONB NOT NULL,
    "errors" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "PerformanceTestResult_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OptimizationRecommendation" (
    "id" TEXT NOT NULL,
    "testResultId" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "impact" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "implementation" TEXT,
    "estimatedImprovement" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "OptimizationRecommendation_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ABTest" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "variantA" TEXT NOT NULL,
    "variantB" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "metric" TEXT NOT NULL,
    "targetImprovement" DOUBLE PRECISION NOT NULL,
    "status" TEXT NOT NULL,
    "startDate" TIMESTAMP(3),
    "endDate" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ABTest_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ABTestResult" (
    "id" TEXT NOT NULL,
    "testId" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "startDate" TIMESTAMP(3) NOT NULL,
    "endDate" TIMESTAMP(3),
    "variantAMetrics" JSONB NOT NULL,
    "variantBMetrics" JSONB NOT NULL,
    "winner" TEXT,
    "improvement" DOUBLE PRECISION,
    "confidenceLevel" DOUBLE PRECISION,
    "notes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ABTestResult_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PerformanceMetric" (
    "id" TEXT NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "system" TEXT NOT NULL,
    "component" TEXT NOT NULL,
    "metric" TEXT NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "unit" TEXT NOT NULL,
    "tags" JSONB,

    CONSTRAINT "PerformanceMetric_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "PerformanceTest_testType_idx" ON "PerformanceTest"("testType");

-- CreateIndex
CREATE INDEX "PerformanceTest_modelId_idx" ON "PerformanceTest"("modelId");

-- CreateIndex
CREATE INDEX "PerformanceTest_strategyId_idx" ON "PerformanceTest"("strategyId");

-- CreateIndex
CREATE INDEX "PerformanceTestResult_testId_idx" ON "PerformanceTestResult"("testId");

-- CreateIndex
CREATE INDEX "PerformanceTestResult_status_idx" ON "PerformanceTestResult"("status");

-- CreateIndex
CREATE INDEX "OptimizationRecommendation_testResultId_idx" ON "OptimizationRecommendation"("testResultId");

-- CreateIndex
CREATE INDEX "OptimizationRecommendation_category_idx" ON "OptimizationRecommendation"("category");

-- CreateIndex
CREATE INDEX "OptimizationRecommendation_impact_idx" ON "OptimizationRecommendation"("impact");

-- CreateIndex
CREATE INDEX "ABTest_type_idx" ON "ABTest"("type");

-- CreateIndex
CREATE INDEX "ABTest_status_idx" ON "ABTest"("status");

-- CreateIndex
CREATE INDEX "ABTestResult_testId_idx" ON "ABTestResult"("testId");

-- CreateIndex
CREATE INDEX "PerformanceMetric_timestamp_idx" ON "PerformanceMetric"("timestamp");

-- CreateIndex
CREATE INDEX "PerformanceMetric_system_component_idx" ON "PerformanceMetric"("system", "component");

-- CreateIndex
CREATE INDEX "PerformanceMetric_metric_idx" ON "PerformanceMetric"("metric");

-- AddForeignKey
ALTER TABLE "PerformanceTestResult" ADD CONSTRAINT "PerformanceTestResult_testId_fkey" FOREIGN KEY ("testId") REFERENCES "PerformanceTest"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OptimizationRecommendation" ADD CONSTRAINT "OptimizationRecommendation_testResultId_fkey" FOREIGN KEY ("testResultId") REFERENCES "PerformanceTestResult"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ABTestResult" ADD CONSTRAINT "ABTestResult_testId_fkey" FOREIGN KEY ("testId") REFERENCES "ABTest"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
