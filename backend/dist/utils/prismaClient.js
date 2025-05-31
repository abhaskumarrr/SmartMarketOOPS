"use strict";
/**
 * Central Prisma Client
 * Singleton to avoid multiple Prisma Client instances
 */
Object.defineProperty(exports, "__esModule", { value: true });
const prisma_1 = require("../../generated/prisma");
// Initialize Prisma Client
const prisma = global.prisma || new prisma_1.PrismaClient();
// Attach to global in non-production environments
if (process.env.NODE_ENV !== 'production') {
    global.prisma = prisma;
}
// Log Prisma initialization
console.log('Prisma Client initialized successfully');
exports.default = prisma;
//# sourceMappingURL=prismaClient.js.map