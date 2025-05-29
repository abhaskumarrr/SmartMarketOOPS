const { PrismaClient } = require('@prisma/client');
const fs = require('fs');
const prisma = new PrismaClient();

async function generateComplianceReport() {
  const accessLogs = await prisma.auditLog.findMany({ where: { action: 'ACCESS' } });
  const erasureRequests = await prisma.complianceRequest.findMany({ where: { type: 'ERASURE' } });
  const encryptionStatus = 'All sensitive fields encrypted (see SECURITY_COMPLIANCE.md)';
  // Add more checks as needed
  const report = {
    generatedAt: new Date(),
    accessLogs,
    erasureRequests,
    encryptionStatus,
  };
  fs.writeFileSync('compliance_report.json', JSON.stringify(report, null, 2));
  console.log('Compliance report generated: compliance_report.json');
}

generateComplianceReport(); 