import express from 'express';

const router = express.Router();

router.get('/dashboard-summary', (req, res) => {
  res.json({
    portfolioValue: (Math.random() * 10000 + 5000).toFixed(2),
    dailyChange: (Math.random() * 5 - 2.5).toFixed(2),
    activePositions: Math.floor(Math.random() * 10),
    profitablePositions: Math.floor(Math.random() * 10),
    aiAccuracy: (Math.random() * 20 + 70).toFixed(2),
  });
});

export default router;
