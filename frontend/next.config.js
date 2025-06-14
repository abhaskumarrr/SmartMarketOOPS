/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: {
    // Disabling ESLint during production builds during development
    // Remove this when the project is ready for production
    ignoreDuringBuilds: true,
  },
  images: {
    domains: ['localhost'],
  },
  // Enable experimental features needed for the project
  experimental: {
    serverActions: true,
  },
}

module.exports = nextConfig 