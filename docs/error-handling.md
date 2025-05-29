# Error Handling in SmartMarketOOPS

This document outlines the error handling approach used in the SmartMarketOOPS application, covering both frontend and backend components.

## Backend Error Handling

### Global Error Handler

The backend uses a centralized error handling approach with the following components:

1. **AppError Class**: A custom error class that includes:
   - `statusCode`: HTTP status code for the error
   - `isOperational`: Flag to indicate whether the error is an expected operational error
   - Stack trace capturing

2. **Global Error Handler Middleware**: Provides consistent error responses:
   - Logs errors for debugging
   - Returns JSON error responses with appropriate status codes
   - Includes stack traces in development mode only

3. **Async Handler**: A utility wrapper for async route handlers to avoid try/catch boilerplate:
   ```typescript
   export const asyncHandler = (fn: Function) => (req: Request, res: Response, next: NextFunction) => {
     Promise.resolve(fn(req, res, next)).catch(next);
   };
   ```

4. **Not Found Handler**: Middleware that catches requests to undefined routes

### Error Categories

Errors are categorized as:

1. **Operational Errors**: Expected errors that can be handled gracefully
   - Invalid user input
   - Authentication/authorization failures
   - Resource not found
   - Rate limiting
   - Network issues

2. **Programming Errors**: Bugs that should be fixed rather than handled at runtime
   - Unhandled exceptions
   - Type errors
   - Syntax errors

### Status Codes

Common HTTP status codes used:

- `200` - OK: Request succeeded
- `201` - Created: Resource created successfully
- `400` - Bad Request: Invalid input
- `401` - Unauthorized: Authentication required
- `403` - Forbidden: Insufficient permissions
- `404` - Not Found: Resource not found
- `409` - Conflict: Resource already exists
- `422` - Unprocessable Entity: Validation error
- `429` - Too Many Requests: Rate limit exceeded
- `500` - Internal Server Error: Unexpected server error

## Frontend Error Handling

### Next.js Error Pages

Custom error pages for different scenarios:

1. **`_error.tsx`**: Custom error component for server-side errors
2. **`404.tsx`**: Custom page for 404 not found errors

### Permission Handling

For authentication and authorization:

1. **PermissionGuard Component**: Conditionally renders content based on user permissions:
   - Displays a loading indicator while checking permissions
   - Shows error alerts when permission checks fail
   - Renders fallback content when the user lacks permissions

2. **usePermission Hook**: Checks if a user has specific permissions:
   - Handles multiple permissions (requireAll or requireAny)
   - Provides loading state for UI feedback
   - Returns error messages for failed permission checks

### API Request Error Handling

For data fetching and API calls:

1. **Try/Catch Blocks**: Wrap all fetch/axios calls
2. **Error States**: Track error messages in component state
3. **Loading States**: Provide user feedback during requests
4. **Default Values**: Fallback to empty arrays/objects when data fetching fails

### Examples

#### Backend Example

```typescript
// Controller using asyncHandler
export const getProfile = asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  if (!req.user) {
    throw new AppError('Not authenticated', 401);
  }

  const user = await prisma.user.findUnique({
    where: { id: req.user.id },
    select: {
      id: true,
      name: true,
      email: true,
      role: true,
      isVerified: true
    }
  });

  if (!user) {
    throw new AppError('User not found', 404);
  }

  res.status(200).json({
    success: true,
    data: user
  });
});
```

#### Frontend Example

```tsx
// React component with error handling
const UserProfile = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { token } = useAuth();

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch('/api/users/profile', {
          headers: { Authorization: `Bearer ${token}` }
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || 'Failed to fetch profile');
        }
        
        const data = await response.json();
        setUser(data.data);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchProfile();
  }, [token]);

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorAlert message={error} />;
  if (!user) return <EmptyState message="No profile data available" />;
  
  return <ProfileCard user={user} />;
};
```

## Testing Error Handling

To ensure proper error handling:

1. **Unit Tests**: Test individual error cases in isolation
2. **Integration Tests**: Verify proper error propagation between components
3. **End-to-End Tests**: Test the entire error flow from UI to backend and back

## Error Logging and Monitoring

The application uses:

1. **Server-side Logging**: Errors are logged to files and console
2. **Client-side Error Tracking**: Console errors in development; could be extended with error tracking services in production
3. **Monitoring**: Server health endpoints provide system status

## Conclusion

This multi-layered approach to error handling provides:

- Better user experience with appropriate feedback
- Easier debugging with detailed error information in development
- Security through information hiding in production
- Consistent error responses across the application 