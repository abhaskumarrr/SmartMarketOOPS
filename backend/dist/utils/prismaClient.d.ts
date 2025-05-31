/**
 * Central Prisma Client
 * Singleton to avoid multiple Prisma Client instances
 */
import { PrismaClient } from '../../generated/prisma';
declare const prisma: PrismaClient<import("../../generated/prisma").Prisma.PrismaClientOptions, never, import("../../generated/prisma/runtime/library").DefaultArgs>;
export default prisma;
