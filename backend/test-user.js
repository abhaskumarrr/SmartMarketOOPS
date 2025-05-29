const { PrismaClient } = require('./generated/prisma');
const bcrypt = require('bcryptjs');

// Initialize Prisma client with explicit connection parameters
const prisma = new PrismaClient({
  datasources: {
    db: {
      url: "postgresql://postgres:postgres@localhost:5432/smoops?schema=public"
    }
  }
});

async function createTestUser() {
  try {
    const hashedPassword = await bcrypt.hash('TestPassword123', 10);
    const user = await prisma.user.create({
      data: {
        name: 'Test User',
        email: 'test@example.com',
        password: hashedPassword,
        isVerified: true
      }
    });
    console.log('Test user created successfully:', user);
  } catch (error) {
    console.error('Error creating test user:', error);
  } finally {
    await prisma.$disconnect();
  }
}

createTestUser(); 