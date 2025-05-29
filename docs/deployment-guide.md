# SMOOPs Deployment Guide

This guide covers deployment options and procedures for the SMOOPs trading bot application.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Environment Configuration](#environment-configuration)
- [Docker Deployment](#docker-deployment)
- [Manual Deployment](#manual-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Security Considerations](#security-considerations)
- [Updating the Application](#updating-the-application)
- [Troubleshooting](#troubleshooting)

## Deployment Options

The SMOOPs trading bot can be deployed in several ways:

1. **Docker Deployment** (Recommended): Deploy using Docker and Docker Compose
2. **Manual Deployment**: Deploy each service individually
3. **Cloud Deployment**: Deploy to cloud providers (AWS, GCP, Azure)

## Environment Configuration

Before deployment, ensure your environment variables are properly configured:

1. Copy `example.env` to `.env` and modify with your production values
2. For production, ensure these variables are set with secure values:
   - `NODE_ENV=production`
   - `ENCRYPTION_MASTER_KEY` (generated with `npm run generate-key`)
   - `DATABASE_URL` (your production database)
   - `DELTA_EXCHANGE_API_KEY` and `DELTA_EXCHANGE_API_SECRET`
   - `DELTA_EXCHANGE_TESTNET=false` (for real trading)

For cloud deployments, use the environment variable management system of your cloud provider rather than `.env` files.

## Docker Deployment

### Prerequisites

- Docker and Docker Compose installed on your server
- Git access to the repository

### Deployment Steps

1. **Clone the repository**

```bash
git clone https://github.com/abhaskumarrr/SMOOPs_dev.git
cd SMOOPs_dev
```

2. **Configure environment variables**

```bash
cp example.env .env
# Edit .env with production values
```

3. **Build and start the Docker containers**

```bash
# Build the images (only needed on first deploy or after image changes)
docker-compose build

# Start the services
docker-compose up -d
```

4. **Verify deployment**

```bash
# Check if all services are running
docker-compose ps

# View logs
docker-compose logs -f
```

### Updating a Docker Deployment

```bash
# Pull the latest code
git pull

# Rebuild the containers if necessary
docker-compose build

# Restart the services
docker-compose up -d
```

## Manual Deployment

For cases where Docker isn't an option or you need more control over each service.

### Backend Deployment

1. **Install Node.js (v18+)**

2. **Clone and set up**

```bash
git clone https://github.com/abhaskumarrr/SMOOPs_dev.git
cd SMOOPs_dev/backend
npm install
```

3. **Set up the database**

```bash
# Install PostgreSQL and create a database

# Configure DATABASE_URL in .env
# Run migrations
npx prisma migrate deploy
```

4. **Build and run**

```bash
npm run build
npm start
```

5. **Process Manager (recommended)**

Use PM2 to manage the Node.js process:

```bash
npm install -g pm2
pm2 start npm --name "smoops-backend" -- start
pm2 save
```

### Frontend Deployment

1. **Install Node.js (v18+)**

2. **Clone and set up**

```bash
git clone https://github.com/abhaskumarrr/SMOOPs_dev.git
cd SMOOPs_dev/frontend
npm install
```

3. **Build the application**

```bash
# Set the API URL in .env.production
echo "NEXT_PUBLIC_API_URL=https://your-api-domain.com" > .env.production
npm run build
```

4. **Deploy with a static server or Next.js server**

```bash
# Option 1: Run with Next.js server
npm start

# Option 2: Export as static site (if no server-side rendering is needed)
npm run export
# Then deploy the 'out' directory to a static hosting service
```

### ML Service Deployment

1. **Install Python 3.10+**

2. **Clone and set up**

```bash
git clone https://github.com/abhaskumarrr/SMOOPs_dev.git
cd SMOOPs_dev/ml
pip install -r requirements.txt
```

3. **Run the service**

```bash
python -m ml.backend.src.scripts.server
```

4. **Process Manager (recommended)**

Use Gunicorn with Supervisor for production:

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker "ml.backend.src.scripts.server:app" --bind 0.0.0.0:3002
```

## Cloud Deployment

### AWS Deployment

The application can be deployed to AWS using several options:

1. **EC2 Instances**
   - Launch EC2 instances for each service
   - Use the manual or Docker deployment steps above
   - Set up an Application Load Balancer

2. **ECS (Elastic Container Service)**
   - Create a `docker-compose.yml` compatible task definition
   - Push Docker images to ECR (Elastic Container Registry)
   - Deploy services to ECS

3. **Serverless Options**
   - Deploy frontend to S3 + CloudFront
   - Deploy backend to Lambda with API Gateway
   - Use Aurora Serverless for PostgreSQL
   - Use SageMaker for ML service

### GCP Deployment

1. **Compute Engine**
   - Similar to AWS EC2 approach

2. **Google Kubernetes Engine (GKE)**
   - Convert Docker Compose to Kubernetes manifests
   - Deploy to GKE cluster

3. **Cloud Run**
   - Containerize each service
   - Deploy to Cloud Run for serverless containers

### Azure Deployment

1. **Virtual Machines**
   - Similar to AWS EC2 approach

2. **Azure Kubernetes Service (AKS)**
   - Convert Docker Compose to Kubernetes manifests
   - Deploy to AKS cluster

3. **Azure App Service**
   - Deploy services as web apps
   - Configure deployment slots for zero-downtime updates

## Monitoring and Maintenance

### Logging

Set up centralized logging with one of these options:

- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **AWS CloudWatch Logs**
- **Google Cloud Logging**
- **Azure Monitor**

### Monitoring

1. **Health Checks**
   - Set up endpoint monitoring for `/health` endpoints
   - Configure alerts for service outages

2. **Performance Monitoring**
   - Use NewRelic, Datadog, or cloud provider monitoring
   - Monitor CPU, memory, database connections
   - Track API response times and error rates

3. **Trading Performance Monitoring**
   - Monitor trading metrics (win rate, profit/loss)
   - Set up alerts for unusual trading activity

### Backups

1. **Database Backups**
   - Set up automated PostgreSQL backups
   - Test backup restoration regularly

2. **Configuration Backups**
   - Back up `.env` files and other configuration
   - Store securely (not in repository)

## Security Considerations

1. **API Security**
   - Use HTTPS for all services
   - Implement rate limiting
   - Use secure HTTP headers

2. **Exchange API Key Security**
   - Use the built-in encryption system for API keys
   - Regularly rotate API keys
   - Use keys with minimal permissions needed

3. **Network Security**
   - Use a firewall to restrict access
   - Configure services to only listen on needed interfaces
   - For cloud: use security groups/VPC

4. **Authentication**
   - Use strong password policies
   - Implement two-factor authentication
   - Consider OAuth integration

## Updating the Application

### Rolling Updates

For Docker deployments:

```bash
# Pull latest code
git pull

# Rebuild specific service
docker-compose build backend

# Update just that service
docker-compose up -d --no-deps backend
```

### Database Migrations

```bash
# Run migrations in production
cd backend
npx prisma migrate deploy
```

## Troubleshooting

### Container Issues

If containers fail to start:

1. Check logs: `docker-compose logs [service]`
2. Verify environment variables are set
3. Check for port conflicts
4. Ensure database is reachable

### Database Connection Issues

1. Verify DB_URL format and credentials
2. Check if database server is running
3. Test connection manually: `psql $DATABASE_URL`
4. Check network/firewall configuration

### Memory/Performance Issues

1. Monitor resource usage: `docker stats`
2. Increase container resource limits if needed
3. Consider database query optimization
4. For ML service: adjust batch sizes, model complexity

---

For additional deployment support, please contact the development team. 