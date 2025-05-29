const express = require('express');
const router = express.Router();
const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

// Data export endpoint
router.post('/export', async (req, res) => {
  const { userId } = req.body;
  const user = await prisma.user.findUnique({ where: { id: userId } });
  // Fetch related data as needed
  res.json({ user });
});

// Data erasure endpoint
router.post('/erase', async (req, res) => {
  const { userId } = req.body;
  await prisma.user.update({ where: { id: userId }, data: { deletedAt: new Date() } });
  // Soft delete related data as needed
  res.json({ status: 'erased' });
});

module.exports = router; 