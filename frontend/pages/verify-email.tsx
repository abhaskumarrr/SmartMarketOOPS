import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import {
  Box,
  Typography,
  Paper,
  Container,
  Alert,
  Button,
  CircularProgress
} from '@mui/material';
import { useAuth } from '../lib/contexts/AuthContext';
import Head from 'next/head';

const VerifyEmailPage: React.FC = () => {
  const [verifying, setVerifying] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const router = useRouter();
  const { verifyEmail, isAuthenticated, user } = useAuth();
  const { token } = router.query;

  // Automatically verify when token is present in URL
  useEffect(() => {
    if (token && typeof token === 'string') {
      handleVerification(token);
    }
  }, [token]);

  // Redirect if already verified
  useEffect(() => {
    if (isAuthenticated && user?.isVerified) {
      router.push('/dashboard');
    }
  }, [isAuthenticated, user, router]);

  const handleVerification = async (verificationToken: string) => {
    setVerifying(true);
    setError('');
    
    try {
      const result = await verifyEmail(verificationToken);
      
      if (result) {
        setSuccess(true);
        // Redirect after a short delay
        setTimeout(() => {
          router.push('/dashboard');
        }, 3000);
      } else {
        setError('Email verification failed. The link may be expired or invalid.');
      }
    } catch (err) {
      console.error('Verification error:', err);
      setError('An error occurred during verification. Please try again.');
    } finally {
      setVerifying(false);
    }
  };

  return (
    <>
      <Head>
        <title>Verify Email | SmartMarket</title>
      </Head>
      <Container maxWidth="sm">
        <Box sx={{ maxWidth: 600, mx: 'auto', my: 8 }}>
          <Paper sx={{ p: 4 }} elevation={3}>
            <Typography variant="h5" component="h1" gutterBottom align="center">
              Email Verification
            </Typography>

            {verifying && (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                <CircularProgress />
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            {success && (
              <Alert severity="success" sx={{ mb: 3 }}>
                Your email has been successfully verified! Redirecting to dashboard...
              </Alert>
            )}

            {!token && !verifying && !success && (
              <>
                <Typography sx={{ mb: 3 }}>
                  Please check your email for a verification link. Click the link to verify your account.
                </Typography>
                
                <Typography sx={{ mb: 3 }}>
                  If you don't see the email, check your spam folder or request a new verification link.
                </Typography>
                
                <Button 
                  variant="contained" 
                  color="primary" 
                  fullWidth
                  onClick={() => router.push('/login')}
                  sx={{ mt: 2 }}
                >
                  Back to Login
                </Button>
              </>
            )}
          </Paper>
        </Box>
      </Container>
    </>
  );
};

export default VerifyEmailPage; 