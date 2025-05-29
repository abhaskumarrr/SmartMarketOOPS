import React from 'react';
import Head from 'next/head';

function MyApp({ Component, pageProps }) {
  return (
    <React.Fragment>
      <Head>
        <title>SmartMarketOOPS Trading Platform</title>
        <meta name="viewport" content="minimum-scale=1, initial-scale=1, width=device-width" />
        <style jsx global>{`
          body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            background-color: #121212;
            color: white;
          }
        `}</style>
      </Head>
      <Component {...pageProps} />
    </React.Fragment>
  );
}

export default MyApp; 