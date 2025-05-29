import React from 'react';
import Head from 'next/head';
import { AppProps } from 'next/app';
import { CacheProvider, EmotionCache } from '@emotion/react';
import createEmotionCache from '../lib/createEmotionCache';
import { ThemeProvider } from '../lib/contexts/ThemeContext';
import { TradingProvider } from '../lib/contexts/TradingContext';
import Layout from '../components/layout/Layout';

// Client-side cache, shared for the whole session of the user in the browser
const clientSideEmotionCache = createEmotionCache();

interface MyAppProps extends AppProps {
  emotionCache?: EmotionCache;
}

function MyApp({ Component, pageProps, emotionCache = clientSideEmotionCache }: MyAppProps) {
  return (
    <CacheProvider value={emotionCache}>
      <Head>
        <title>SmartMarketOOPS Trading Platform</title>
        <meta name="viewport" content="minimum-scale=1, initial-scale=1, width=device-width" />
        <meta name="description" content="Advanced trading platform with real-time market data and ML-powered signals" />
        <link rel="icon" href="/favicon.ico" />
        {/* Load Inter font */}
        <link 
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" 
          rel="stylesheet"
        />
      </Head>
      <ThemeProvider>
        <TradingProvider>
          <Layout>
            <Component {...pageProps} />
          </Layout>
        </TradingProvider>
      </ThemeProvider>
    </CacheProvider>
  );
}

export default MyApp; 