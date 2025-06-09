# Use a Node.js 20 base image
FROM node:20

WORKDIR /app

# Clean up previous builds and node_modules
RUN rm -rf node_modules dist

# Install dependencies
COPY package*.json ./
RUN npm install --include=dev
ENV PATH="/app/node_modules/.bin:$PATH"

# Copy source code
COPY . .

# Generate Prisma client (will be done at runtime if needed)
# RUN npx prisma generate

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 --ingroup nodejs --home /app nodejs
USER nodejs

# Expose API port
EXPOSE 3006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3006/health || exit 1

# Start the server - using ts-node to compile and run in one step
CMD ["npx", "ts-node", "--transpile-only", "src/server.ts"]
