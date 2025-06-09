# Preventing Version Skew Errors in Next.js

This document explains how we resolved the `TypeError: Cannot read properties of undefined (reading 'call')` errors in our Next.js application.

## Understanding the Problem

These errors occur due to **version skew** - when different parts of the application are running different versions of code, particularly during deployments or when multiple versions of the same module are loaded.

Common causes:
- Deploying new code while old code is still active
- Different versions of dynamically loaded modules
- React Server Components (RSC) using mismatched client/server code

## Solutions Implemented

We've implemented multiple layers of protection:

### 1. Version Skew Protection Middleware

```typescript
// src/middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Force MPA navigation when version mismatch is detected
export function middleware(request: NextRequest) {
  // Check RSC requests and deployment ID cookies
  if (request.headers.get('accept')?.includes('text/x-component')) {
    // Version skew protection logic
  }
}
```

### 2. Client-Side Error Detection

```typescript
// src/components/VersionSkewProtection.tsx
'use client';

export function VersionSkewProtection({ children }) {
  useEffect(() => {
    // Override console.error to catch specific webpack errors
    console.error = (...args) => {
      if (errorMessage.includes('TypeError: Cannot read properties of undefined (reading \'call\')')) {
        // Force page refresh
      }
    };
  }, []);
}
```

### 3. Deployment ID Tracking

- Added `NEXT_DEPLOYMENT_ID` environment variable
- Used in both server and client to detect mismatches
- Updated on each deployment with unique timestamp

### 4. Next.js Configuration

```typescript
// next.config.ts
const nextConfig = {
  experimental: {
    // Improved module resolution
    taint: true,
    // Server Actions configuration
    serverActions: { bodySizeLimit: '2mb' },
  },
  webpack: (config) => {
    // Add version hash to modules
    config.output.chunkLoadingGlobal = `webpackChunk_${Date.now()}`;
    return config;
  },
};
```

### 5. Blue-Green Deployment Strategy

- Created `docker-compose.prod.yml` with blue-green deployment settings
- Used Nginx as a proxy to handle version transitions
- Added deployment script that handles versioning

## How to Use

1. **During Development:**
   - The `VersionSkewProtection` component is already applied in `layout.tsx`
   - No additional steps needed during development

2. **For Deployments:**
   - Use the `./deploy.sh` script which:
     - Generates a unique deployment ID
     - Builds Docker images with consistent versioning
     - Handles blue-green deployment

3. **Monitoring:**
   - If version skew errors occur, users will see a "Updating application..." message
   - The page will automatically refresh to get the latest version

## Additional Resources

- [Next.js Deployment Documentation](https://nextjs.org/docs/deployment)
- [Vercel Skew Protection](https://vercel.com/docs/deployments/skew-protection)
- [React Server Components RFC](https://github.com/reactjs/rfcs/blob/main/text/0188-server-components.md)

## Related GitHub Issues

- [NextJS 14.1.0 TypeError: Cannot read properties of undefined (reading 'call')](https://github.com/vercel/next.js/issues/61995) 