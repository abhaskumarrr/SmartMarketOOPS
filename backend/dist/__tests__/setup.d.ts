/**
 * Jest Test Setup
 * Global setup for all tests
 */
declare global {
    namespace jest {
        interface Matchers<R> {
            toBeValidDate(): R;
            toBeValidUUID(): R;
            toBeWithinRange(floor: number, ceiling: number): R;
        }
    }
    var testUtils: {
        generateMockUser: () => any;
        generateMockBot: () => any;
        generateMockMarketData: () => any;
        sleep: (ms: number) => Promise<void>;
        randomString: (length?: number) => string;
        randomEmail: () => string;
        mockApiSuccess: (data: any) => any;
        mockApiError: (message?: string) => any;
    };
}
export {};
