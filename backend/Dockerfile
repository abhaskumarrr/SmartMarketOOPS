# syntax=docker/dockerfile:1
FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Generate Prisma client
RUN npx prisma generate

# Build for production (if needed)
# RUN npm run build

# Expose API port
EXPOSE 3001

# Start the server
CMD ["npm", "run", "dev"]
