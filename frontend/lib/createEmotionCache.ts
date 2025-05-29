import createCache from '@emotion/cache';

// On the client side, Create a custom cache
export default function createEmotionCache() {
  return createCache({ key: 'css', prepend: true });
} 