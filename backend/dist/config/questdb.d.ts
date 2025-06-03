/**
 * QuestDB Configuration
 * High-performance time-series database configuration for SmartMarketOOPS
 */
import { Sender } from '@questdb/nodejs-client';
export interface QuestDBConfig {
    host: string;
    port: number;
    username?: string;
    password?: string;
    database?: string;
    ssl?: boolean;
    connectionTimeout?: number;
    queryTimeout?: number;
    maxConnections?: number;
    retryAttempts?: number;
    retryDelay?: number;
}
export interface QuestDBConnectionOptions {
    host?: string;
    port?: number;
    username?: string;
    password?: string;
    token?: string;
    tls?: boolean;
    autoFlush?: boolean;
    autoFlushRows?: number;
    autoFlushInterval?: number;
    requestMinThroughput?: number;
    requestTimeout?: number;
    retryTimeout?: number;
    maxBufferSize?: number;
}
declare const defaultConfig: QuestDBConfig;
declare const defaultClientOptions: QuestDBConnectionOptions;
export declare class QuestDBConnection {
    private static instance;
    private client;
    private config;
    private clientOptions;
    private isConnected;
    private connectionPromise;
    private constructor();
    static getInstance(): QuestDBConnection;
    connect(): Promise<void>;
    private _connect;
    disconnect(): Promise<void>;
    getClient(): Sender;
    isReady(): boolean;
    getConfig(): QuestDBConfig;
    updateConfig(newConfig: Partial<QuestDBConfig>): void;
    updateClientOptions(newOptions: Partial<QuestDBConnectionOptions>): void;
    healthCheck(): Promise<boolean>;
    getStats(): {
        isConnected: boolean;
        config: QuestDBConfig;
        clientOptions: QuestDBConnectionOptions;
    };
}
export declare const questdbConnection: QuestDBConnection;
export { defaultConfig as questdbConfig, defaultClientOptions as questdbClientOptions };
export declare function createQuestDBClient(options?: Partial<QuestDBConnectionOptions>): Sender;
export declare function validateQuestDBEnvironment(): {
    valid: boolean;
    errors: string[];
};
export declare function connectWithRetry(maxAttempts?: number, delay?: number): Promise<void>;
