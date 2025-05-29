import React from 'react';
import { Box, Typography, Button, Container, Paper } from '@mui/material';
import { useRouter } from 'next/router';
import Head from 'next/head';

function NotFoundPage() {
  const router = useRouter();

  return (
    <>
      <Head>
        <title>404 - Page Not Found | SmartMarketOOPS</title>
      </Head>
      <Container maxWidth="md" sx={{ mt: 10 }}>
        <Paper 
          elevation={3} 
          sx={{ 
            p: 5, 
            textAlign: 'center',
            borderRadius: 2,
            backgroundColor: theme => theme.palette.background.paper
          }}
        >
          <Typography variant="h1" component="h1" gutterBottom sx={{ fontSize: '5rem', fontWeight: 700 }}>
            404
          </Typography>
          <Typography variant="h5" component="h2" gutterBottom color="text.secondary">
            The page you are looking for does not exist.
          </Typography>
          <Box sx={{ mt: 4 }}>
            <Button 
              variant="contained" 
              color="primary" 
              size="large"
              onClick={() => router.push('/')}
              sx={{ mr: 2 }}
            >
              Go to Home
            </Button>
            <Button 
              variant="outlined" 
              color="primary" 
              size="large"
              onClick={() => router.back()}
            >
              Go Back
            </Button>
          </Box>
        </Paper>
      </Container>
    </>
  );
}

export default NotFoundPage; 