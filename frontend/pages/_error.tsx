import React from 'react';
import { Box, Typography, Button, Container, Paper } from '@mui/material';
import { useRouter } from 'next/router';
import Head from 'next/head';
import type { NextPageContext } from 'next';

interface ErrorProps {
  statusCode: number;
  message?: string;
}

function ErrorPage({ statusCode, message }: ErrorProps) {
  const router = useRouter();

  // Default error messages based on status code
  const getErrorMessage = (): string => {
    if (statusCode === 404) {
      return message || 'The page you are looking for does not exist.';
    } else if (statusCode === 500) {
      return message || 'An internal server error occurred.';
    } else if (statusCode === 403) {
      return message || 'You do not have permission to access this page.';
    } else if (statusCode === 401) {
      return message || 'Authentication is required to access this page.';
    }
    return message || 'An error occurred.';
  };

  return (
    <>
      <Head>
        <title>Error {statusCode} | SmartMarketOOPS</title>
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
            {statusCode || 'Error'}
          </Typography>
          <Typography variant="h5" component="h2" gutterBottom color="text.secondary">
            {getErrorMessage()}
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

ErrorPage.getInitialProps = ({ res, err }: NextPageContext): ErrorProps => {
  const statusCode = res ? res.statusCode : err ? err.statusCode || 500 : 404;
  return { statusCode };
};

export default ErrorPage; 