import { PrismaClient } from '@prisma/client';

/**
 * Prisma client for read-only database operations.
 * Connects to the read replica database.
 */
const prismaReadOnly = new PrismaClient({
  datasources: {
    db: {
      url: process.env.READ_ONLY_DATABASE_URL,
    },
  },
});

export default prismaReadOnly; 