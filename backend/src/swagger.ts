/**
 * Swagger API Documentation Setup
 * 
 * This file configures Swagger UI for the SmartMarketOOPS API documentation.
 * It serves the OpenAPI specification at /api-docs endpoint.
 */

import express from 'express';
import swaggerUi from 'swagger-ui-express';
import swaggerDocument from './swagger.json';
import { logger } from './utils/logger';

/**
 * Initialize Swagger API documentation
 * @param app Express application instance
 */
export const setupSwagger = (app: express.Application): void => {
  try {
    // Serve Swagger UI at /api-docs endpoint
    app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument, {
      explorer: true,
      customCss: '.swagger-ui .topbar { display: none }',
      customSiteTitle: 'SmartMarketOOPS API Documentation',
      swaggerOptions: {
        persistAuthorization: true,
        tagsSorter: 'alpha',
        operationsSorter: 'alpha',
      },
    }));

    // Serve raw OpenAPI spec at /api-docs.json endpoint
    app.get('/api-docs.json', (req, res) => {
      res.setHeader('Content-Type', 'application/json');
      res.send(swaggerDocument);
    });
    
    logger.info('📚 Swagger API documentation initialized at /api-docs');
  } catch (error) {
    logger.error('Failed to initialize Swagger API documentation:', error);
  }
}; 