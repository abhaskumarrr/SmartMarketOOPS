import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  Alert,
  CircularProgress,
  Container,
  Grid,
} from '@mui/material';
import { useAuth } from '../lib/contexts/AuthContext';
import Head from 'next/head';

const ResetPasswordPage: React.FC = () => {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [tokenError, setTokenError] = useState(false);

  const { resetPassword } = useAuth();
  const router = useRouter();
  const { token } = router.query;

  // Validate token presence
  useEffect(() => {
    if (router.isReady && (!token || typeof token !== 'string')) {
      setTokenError(true);
    }
  }, [router.isReady, token]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      // Validate token
      if (!token || typeof token !== 'string') {
        setError('Invalid or missing reset token. Please request a new password reset link.');
        setLoading(false);
        return;
      }

      // Validate passwords
      if (!password || !confirmPassword) {
        setError('Please enter and confirm your new password');
        setLoading(false);
        return;
      }

      if (password !== confirmPassword) {
        setError('Passwords do not match');
        setLoading(false);
        return;
      }

      if (password.length < 8) {
        setError('Password must be at least 8 characters long');
        setLoading(false);
        return;
      }

      const result = await resetPassword(token, password);

      if (result) {
        setSuccess(true);
        // Redirect after a short delay
        setTimeout(() => {
          router.push('/login');
        }, 3000);
      } else {
        setError('Failed to reset password. The link may be expired or invalid.');
      }
    } catch (err) {
      setError('An error occurred. Please try again later.');
      console.error('Password reset error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Head>
        <title>Reset Password | SmartMarket</title>
      </Head>
      <Container maxWidth="sm">
        <Box sx={{ maxWidth: 450, mx: 'auto', my: 8 }}>
          <Paper sx={{ p: 4 }} elevation={3}>
            <Typography variant="h5" component="h1" gutterBottom align="center">
              Reset Your Password
            </Typography>

            {tokenError ? (
              <>
                <Alert severity="error" sx={{ mb: 3 }}>
                  Invalid or missing reset token. Please request a new password reset link.
                </Alert>
                <Button
                  fullWidth
                  variant="contained"
                  onClick={() => router.push('/forgot-password')}
                  sx={{ mt: 2 }}
                >
                  Request New Reset Link
                </Button>
              </>
            ) : (
              <>
                {error && (
                  <Alert severity="error" sx={{ mb: 3 }}>
                    {error}
                  </Alert>
                )}

                {success ? (
                  <>
                    <Alert severity="success" sx={{ mb: 3 }}>
                      Your password has been successfully reset! Redirecting to login...
                    </Alert>
                    <Button
                      fullWidth
                      variant="contained"
                      onClick={() => router.push('/login')}
                      sx={{ mt: 2 }}
                    >
                      Go to Login
                    </Button>
                  </>
                ) : (
                  <Box component="form" onSubmit={handleSubmit} noValidate>
                    <Typography sx={{ mb: 3 }}>
                      Please enter and confirm your new password.
                    </Typography>
                    
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <Box>
                        <TextField
                          margin="normal"
                          required
                          fullWidth
                          name="password"
                          label="New Password"
                          type="password"
                          id="password"
                          autoComplete="new-password"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          disabled={loading}
                          helperText="Minimum 8 characters"
                        />
                      </Box>
                      <Box>
                        <TextField
                          margin="normal"
                          required
                          fullWidth
                          name="confirmPassword"
                          label="Confirm New Password"
                          type="password"
                          id="confirmPassword"
                          autoComplete="new-password"
                          value={confirmPassword}
                          onChange={(e) => setConfirmPassword(e.target.value)}
                          disabled={loading}
                        />
                      </Box>
                    </Box>

                    <Button
                      type="submit"
                      fullWidth
                      variant="contained"
                      sx={{ mt: 3, mb: 2 }}
                      disabled={loading}
                    >
                      {loading ? <CircularProgress size={24} /> : 'Reset Password'}
                    </Button>
                  </Box>
                )}
              </>
            )}
          </Paper>
        </Box>
      </Container>
    </>
  );
};

export default ResetPasswordPage; 