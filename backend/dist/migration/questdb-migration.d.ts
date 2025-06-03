/**
 * QuestDB Migration Pipeline
 * Migrates time-series data from PostgreSQL to QuestDB for performance optimization
 */
import { QuestDBMigrationConfig, QuestDBMigrationProgress, QuestDBMigrationResult } from '../types/questdb';
export declare class QuestDBMigrationPipeline {
    private prisma;
    private validator;
    private migrationProgress;
    constructor();
    /**
     * Initialize the migration pipeline
     */
    initialize(): Promise<void>;
    /**
     * Shutdown the migration pipeline
     */
    shutdown(): Promise<void>;
    /**
     * Migrate all time-series tables
     */
    migrateAll(config?: Partial<QuestDBMigrationConfig>): Promise<QuestDBMigrationResult[]>;
    /**
     * Migrate Metrics table
     */
    migrateMetrics(config: QuestDBMigrationConfig): Promise<QuestDBMigrationResult>;
    /**
     * Migrate TradingSignals table
     */
    migrateTradingSignals(config: QuestDBMigrationConfig): Promise<QuestDBMigrationResult>;
    /**
     * Migrate MLPredictions table
     */
    migrateMLPredictions(config: QuestDBMigrationConfig): Promise<QuestDBMigrationResult>;
    /**
     * Migrate PerformanceMetrics table
     */
    migratePerformanceMetrics(config: QuestDBMigrationConfig): Promise<QuestDBMigrationResult>;
    /**
     * Get migration progress for a table
     */
    getMigrationProgress(tableName: string): QuestDBMigrationProgress | null;
    /**
     * Get migration progress for all tables
     */
    getAllMigrationProgress(): Record<string, QuestDBMigrationProgress>;
    private buildDateFilter;
    private updateProgress;
    private calculateETA;
}
