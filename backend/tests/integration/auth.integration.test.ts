/**
 * Authentication Integration Tests
 * End-to-end tests for the complete authentication flow
 */

import request from 'supertest';
import { PrismaClient } from '@prisma/client';
import { app } from '../../src/server'; // Importing from server.ts where the app is properly exported
import { generateToken } from '../../src/utils/jwt';

// Get a real Prisma client for integration tests
const prisma = new PrismaClient();

describe('Authentication Integration', () => {
  // Test user data
  const testUser = {
    name: 'Integration Test User',
    email: 'integration-test@example.com',
    password: 'SecurePassword123!',
    id: ''
  };
  
  // Tokens that will be set during tests
  let authToken: string;
  let refreshToken: string;
  
  // Before all tests, set up the database
  beforeAll(async () => {
    // Clean up any previous test data
    await prisma.user.deleteMany({
      where: {
        email: testUser.email
      }
    });
  });
  
  // After all tests, clean up
  afterAll(async () => {
    await prisma.user.deleteMany({
      where: {
        email: testUser.email
      }
    });
    
    await prisma.$disconnect();
  });

  describe('Registration Flow', () => {
    it('should register a new user', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          name: testUser.name,
          email: testUser.email,
          password: testUser.password
        });
      
      expect(response.status).toBe(201);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toHaveProperty('id');
      expect(response.body.data).toHaveProperty('token');
      expect(response.body.data).toHaveProperty('refreshToken');
      expect(response.body.data.email).toBe(testUser.email);
      expect(response.body.data.name).toBe(testUser.name);
      expect(response.body.data.isVerified).toBe(false);
      
      // Save the user ID for later tests
      testUser.id = response.body.data.id;
    });
    
    it('should not register a duplicate user', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          name: testUser.name,
          email: testUser.email,
          password: testUser.password
        });
      
      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
      expect(response.body.message).toContain('User already exists');
    });
    
    it('should reject registration with invalid data', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          name: 'Missing Email User',
          // Missing email
          password: 'Password123!'
        });
      
      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
    });
  });

  describe('Login Flow', () => {
    it('should login with valid credentials', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: testUser.email,
          password: testUser.password
        });
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toHaveProperty('token');
      expect(response.body.data).toHaveProperty('refreshToken');
      expect(response.body.data).toHaveProperty('user');
      expect(response.body.data.user.email).toBe(testUser.email);
      
      // Save tokens for later tests
      authToken = response.body.data.token;
      refreshToken = response.body.data.refreshToken;
    });
    
    it('should reject login with invalid credentials', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: testUser.email,
          password: 'WrongPassword123!'
        });
      
      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
      expect(response.body.message).toContain('Invalid credentials');
    });
    
    it('should reject login for non-existent user', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'nonexistent@example.com',
          password: 'Password123!'
        });
      
      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
    });
  });

  describe('Token Verification', () => {
    it('should access protected route with valid token', async () => {
      // Assuming you have a protected route that returns user profile
      const response = await request(app)
        .get('/api/users/profile')
        .set('Authorization', `Bearer ${authToken}`);
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toHaveProperty('email', testUser.email);
    });
    
    it('should reject access to protected route without token', async () => {
      const response = await request(app)
        .get('/api/users/profile');
      
      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
    });
    
    it('should reject access with invalid token', async () => {
      const response = await request(app)
        .get('/api/users/profile')
        .set('Authorization', 'Bearer invalid.token.value');
      
      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
    });
  });

  describe('Refresh Token Flow', () => {
    it('should issue new tokens with valid refresh token', async () => {
      const response = await request(app)
        .post('/api/auth/refresh-token')
        .send({ refreshToken });
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toHaveProperty('token');
      expect(response.body.data).toHaveProperty('refreshToken');
      
      // Update the tokens for subsequent tests
      authToken = response.body.data.token;
      refreshToken = response.body.data.refreshToken;
    });
    
    it('should reject invalid refresh token', async () => {
      const response = await request(app)
        .post('/api/auth/refresh-token')
        .send({ refreshToken: 'invalid.refresh.token' });
      
      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
    });
  });

  describe('Password Reset Flow', () => {
    let resetToken: string;
    
    it('should send password reset email', async () => {
      const response = await request(app)
        .post('/api/auth/forgot-password')
        .send({ email: testUser.email });
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      
      // In a real test, we would have to capture the email, but for now
      // we can mock this by getting the reset token directly from the database
      const user = await prisma.user.findUnique({
        where: { email: testUser.email }
      });
      
      resetToken = user?.resetPasswordToken || '';
      expect(resetToken).toBeTruthy();
    });
    
    it('should reset password with valid token', async () => {
      const newPassword = 'NewSecurePassword456!';
      
      const response = await request(app)
        .post('/api/auth/reset-password')
        .send({
          token: resetToken,
          password: newPassword
        });
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      
      // Verify we can login with the new password
      const loginResponse = await request(app)
        .post('/api/auth/login')
        .send({
          email: testUser.email,
          password: newPassword
        });
      
      expect(loginResponse.status).toBe(200);
      expect(loginResponse.body.success).toBe(true);
      
      // Update the password for future tests
      testUser.password = newPassword;
    });
    
    it('should reject invalid reset token', async () => {
      const response = await request(app)
        .post('/api/auth/reset-password')
        .send({
          token: 'invalid-token',
          password: 'AnotherPassword789!'
        });
      
      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
    });
  });

  describe('Logout Flow', () => {
    it('should successfully logout the user', async () => {
      const response = await request(app)
        .post('/api/auth/logout')
        .set('Authorization', `Bearer ${authToken}`);
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      
      // Verify token is no longer valid by trying to access a protected route
      const profileResponse = await request(app)
        .get('/api/users/profile')
        .set('Authorization', `Bearer ${authToken}`);
      
      // The exact behavior depends on your implementation:
      // Some systems invalidate the token in a blacklist, others rely on short expiration
      // If your system uses a token blacklist, this should be 401
      // If your system relies only on token expiration, this might still work until the token expires
      
      // For this test, we'll assume the token is immediately invalidated
      expect(profileResponse.status).toBe(401);
    });
  });
}); 