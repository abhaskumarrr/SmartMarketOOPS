import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// The deployment ID helps identify different versions of the app
const DEPLOYMENT_ID = process.env.NEXT_DEPLOYMENT_ID || process.env.VERCEL_DEPLOYMENT_ID || Date.now().toString();

export function middleware(request: NextRequest) {
  const response = NextResponse.next();
  
  // Set the deployment ID as a header
  response.headers.set('x-deployment-id', DEPLOYMENT_ID);
  
  // Check if this is an RSC request
  if (request.headers.get('accept')?.includes('text/x-component')) {
    const clientDeployId = request.cookies.get('deployment-id')?.value;
    
    // If the client has a different deployment ID, force a full page refresh
    if (clientDeployId && clientDeployId !== DEPLOYMENT_ID) {
      console.log('Version skew detected, forcing full page refresh');
      
      // Return plain text to force client to do a full page refresh
      return new NextResponse('Version skew detected', {
        status: 200,
        headers: {
          'Content-Type': 'text/plain',
        },
      });
    }
  }
  
  // Set the deployment ID cookie for future requests
  response.cookies.set('deployment-id', DEPLOYMENT_ID, {
    path: '/',
    sameSite: 'strict',
    httpOnly: true,
  });
  
  return response;
}

export const config = {
  // Only run middleware for navigation and RSC requests
  matcher: [
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}; 