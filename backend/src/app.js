const express = require('express');
const complianceRouter = require('./routes/compliance');

const app = express();

// Middleware
app.use(express.json());

// Routes
app.use('/api/compliance', complianceRouter);

module.exports = app; 