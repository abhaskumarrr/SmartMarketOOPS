import React, { useState } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  Alert,
  CircularProgress,
  Link as MuiLink,
  Container,
} from '@mui/material';
import { useAuth } from '../lib/contexts/AuthContext';
import Head from 'next/head';

const ForgotPasswordPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const { forgotPassword } = useAuth();
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      // Basic validation
      if (!email) {
        setError('Please enter your email address');
        setLoading(false);
        return;
      }

      const result = await forgotPassword(email);

      if (result) {
        setSuccess(true);
      } else {
        setError('Failed to process your request. Please try again.');
      }
    } catch (err) {
      setError('An error occurred. Please try again later.');
      console.error('Forgot password error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Head>
        <title>Forgot Password | SmartMarket</title>
      </Head>
      <Container maxWidth="sm">
        <Box sx={{ maxWidth: 450, mx: 'auto', my: 8 }}>
          <Paper sx={{ p: 4 }} elevation={3}>
            <Typography variant="h5" component="h1" gutterBottom align="center">
              Reset Your Password
            </Typography>

            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            {success ? (
              <>
                <Alert severity="success" sx={{ mb: 3 }}>
                  Password reset instructions have been sent to your email.
                </Alert>
                <Typography sx={{ mb: 3 }}>
                  Please check your email for instructions to reset your password. If you don't see the email, check your spam folder.
                </Typography>
                <Button
                  fullWidth
                  variant="contained"
                  onClick={() => router.push('/login')}
                  sx={{ mt: 2 }}
                >
                  Back to Login
                </Button>
              </>
            ) : (
              <Box component="form" onSubmit={handleSubmit} noValidate>
                <Typography sx={{ mb: 3 }}>
                  Enter your email address and we'll send you instructions to reset your password.
                </Typography>
                
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="email"
                  label="Email Address"
                  name="email"
                  autoComplete="email"
                  autoFocus
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={loading}
                />

                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  sx={{ mt: 3, mb: 2 }}
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Send Reset Link'}
                </Button>

                <Box sx={{ mt: 2, textAlign: 'center' }}>
                  <Link href="/login" passHref>
                    <MuiLink variant="body2">
                      Back to Login
                    </MuiLink>
                  </Link>
                </Box>
              </Box>
            )}
          </Paper>
        </Box>
      </Container>
    </>
  );
};

export default ForgotPasswordPage; 