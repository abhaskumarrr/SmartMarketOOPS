import { useRouter } from 'next/router';
import { useEffect } from 'react';
import { useAuth } from '../../lib/contexts/AuthContext';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requireVerified?: boolean;
  requiredRoles?: string[];
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requireVerified = false,
  requiredRoles = [],
}) => {
  const { isAuthenticated, user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    // Don't redirect while the auth state is loading
    if (loading) return;

    // If not authenticated, redirect to login
    if (!isAuthenticated) {
      router.push({
        pathname: '/login',
        query: { returnUrl: router.asPath },
      });
      return;
    }

    // If email verification is required but the user is not verified
    if (requireVerified && user && !user.isVerified) {
      router.push('/verify-email');
      return;
    }

    // If specific roles are required but the user doesn't have them
    if (requiredRoles.length > 0 && user && !requiredRoles.includes(user.role)) {
      router.push('/unauthorized');
      return;
    }
  }, [isAuthenticated, loading, requireVerified, requiredRoles, router, user]);

  // Show nothing while authentication is loading or redirect is happening
  if (loading || !isAuthenticated || (requireVerified && user && !user.isVerified) || 
      (requiredRoles.length > 0 && user && !requiredRoles.includes(user.role))) {
    return <div>Loading...</div>;
  }

  return <>{children}</>;
};

export default ProtectedRoute; 