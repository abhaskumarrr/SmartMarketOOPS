import React, { useEffect, useState } from 'react';
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

const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { login, isAuthenticated } = useAuth();
  const router = useRouter();
  const returnUrl = router.query.returnUrl as string || '/dashboard';

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      router.push(returnUrl);
    }
  }, [isAuthenticated, router, returnUrl]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      // Basic validation
      if (!email || !password) {
        setError('Please enter both email and password');
        setLoading(false);
        return;
      }

      const success = await login(email, password);

      if (success) {
        router.push(returnUrl);
      } else {
        setError('Invalid email or password');
      }
    } catch (err) {
      setError('Login failed. Please try again.');
      console.error('Login error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Head>
        <title>Login | SmartMarket</title>
      </Head>
      <Container maxWidth="sm">
        <Box sx={{ maxWidth: 450, mx: 'auto', my: 8 }}>
          <Paper sx={{ p: 4 }} elevation={3}>
            <Typography variant="h5" component="h1" gutterBottom align="center">
              Log In to SmartMarket
            </Typography>

            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            <Box component="form" onSubmit={handleSubmit} noValidate>
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
              
              <TextField
                margin="normal"
                required
                fullWidth
                name="password"
                label="Password"
                type="password"
                id="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
              />

              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2 }}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Log In'}
              </Button>

              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
                <Link href="/forgot-password" passHref>
                  <MuiLink variant="body2">Forgot password?</MuiLink>
                </Link>
                
                <Link href="/register" passHref>
                  <MuiLink variant="body2">
                    {"Don't have an account? Sign Up"}
                  </MuiLink>
                </Link>
              </Box>
            </Box>
          </Paper>
        </Box>
      </Container>
    </>
  );
};

export default LoginPage; 