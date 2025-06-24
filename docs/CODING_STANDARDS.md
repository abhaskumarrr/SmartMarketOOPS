# SmartMarketOOPS Coding Standards

## Overview

This document outlines the coding standards and best practices for the SmartMarketOOPS project. Following these standards ensures consistency, maintainability, and quality across the codebase.

## Code Style

### TypeScript/JavaScript

We use ESLint and Prettier to enforce code style. The configuration is in:
- `.eslintrc.js` - ESLint configuration
- `.prettierrc` - Prettier configuration

Key style rules:
- Use 2 spaces for indentation
- Use single quotes for strings
- Use semicolons at the end of statements
- Maximum line length of 100 characters
- Use camelCase for variables and functions
- Use PascalCase for classes and interfaces
- Use UPPER_CASE for constants

### Running Linting and Formatting

```bash
# Check for linting issues
npm run lint

# Fix linting issues automatically
npm run lint:fix

# Format code
npm run format

# Check formatting
npm run format:check
```

## Project Structure

```
backend/
├── src/
│   ├── config/         # Configuration files
│   ├── controllers/    # API controllers
│   ├── middleware/     # Express middleware
│   ├── routes/         # API routes
│   ├── services/       # Business logic
│   ├── types/          # TypeScript type definitions
│   ├── utils/          # Utility functions
│   └── server.ts       # Main server entry point
├── tests/              # Test files
└── prisma/             # Database schema and migrations
```

## Naming Conventions

### Files and Directories

- Use kebab-case for file and directory names (e.g., `api-key-controller.ts`)
- Use descriptive names that reflect the purpose of the file
- Group related files in directories

### Code Elements

- **Interfaces**: Prefix with `I` (e.g., `IUser`)
- **Types**: Use descriptive names (e.g., `UserRole`)
- **Enums**: Use PascalCase (e.g., `LogLevel`)
- **Constants**: Use UPPER_CASE (e.g., `MAX_RETRY_COUNT`)

## Imports

- Group imports in the following order:
  1. Node.js built-in modules
  2. External dependencies
  3. Internal modules
- Sort imports alphabetically within each group
- Use absolute imports for internal modules

Example:
```typescript
// Node.js built-in modules
import fs from 'fs';
import path from 'path';

// External dependencies
import express from 'express';
import { v4 as uuidv4 } from 'uuid';

// Internal modules
import { logger } from '../utils/logger';
import { User } from '../types/user';
```

## Error Handling

Use the centralized error handling pattern:

```typescript
import { ApiError, asyncHandler } from '../utils/errorHandler';

export const getUser = asyncHandler(async (req, res) => {
  const user = await userService.findById(req.params.id);
  
  if (!user) {
    throw new ApiError('User not found', 404, true, 'USER_NOT_FOUND');
  }
  
  res.json({ success: true, data: user });
});
```

## Environment Variables

Use the centralized environment configuration:

```typescript
import env from '../config/environment';

const port = env.PORT;
const apiUrl = env.API_URL;
```

## Logging

Use the logger utility for consistent logging:

```typescript
import { createLogger } from '../utils/logger';

const logger = createLogger('UserService');

logger.info('User created successfully', { userId: user.id });
logger.error('Failed to create user', { error: error.message });
```

## Comments and Documentation

- Use JSDoc comments for functions, classes, and interfaces
- Keep comments up-to-date with code changes
- Focus on explaining "why" rather than "what"

Example:
```typescript
/**
 * Creates a new user in the system
 * 
 * @param userData - The user data to create
 * @returns The created user
 * @throws ApiError if validation fails
 */
async function createUser(userData: UserCreateDto): Promise<User> {
  // Implementation
}
```

## Testing

- Write unit tests for all business logic
- Write integration tests for API endpoints
- Use descriptive test names that explain the expected behavior
- Follow the AAA pattern (Arrange, Act, Assert)

Example:
```typescript
describe('UserService', () => {
  describe('createUser', () => {
    it('should create a user when valid data is provided', async () => {
      // Arrange
      const userData = { name: 'Test User', email: 'test@example.com' };
      
      // Act
      const user = await userService.createUser(userData);
      
      // Assert
      expect(user).toHaveProperty('id');
      expect(user.name).toBe(userData.name);
      expect(user.email).toBe(userData.email);
    });
  });
});
```

## Security Best Practices

- Never hardcode credentials or secrets
- Use environment variables for sensitive information
- Validate and sanitize all user input
- Implement proper authentication and authorization
- Use HTTPS for all API calls
- Implement rate limiting for API endpoints
- Use parameterized queries to prevent SQL injection
- Set proper security headers

## Performance Considerations

- Use caching for frequently accessed data
- Optimize database queries
- Use connection pooling
- Implement pagination for large data sets
- Use compression middleware
- Monitor response times

## Git Workflow

- Use feature branches for new features
- Use descriptive branch names (e.g., `feature/add-user-authentication`)
- Write clear commit messages that explain the purpose of the change
- Keep commits focused on a single change
- Squash commits before merging
- Use pull requests for code review

## Conclusion

Following these coding standards ensures that the SmartMarketOOPS codebase remains clean, maintainable, and secure. All team members should adhere to these standards to maintain consistency across the project.