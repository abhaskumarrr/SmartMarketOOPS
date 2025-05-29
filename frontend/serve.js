const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3000;

// Create the server
const server = http.createServer((req, res) => {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  
  // Always serve index.html for any request
  const filePath = path.join(__dirname, 'public', 'index.html');
  
  // Read the file
  fs.readFile(filePath, (error, content) => {
    if (error) {
      if (error.code === 'ENOENT') {
        // File not found
        res.writeHead(404);
        res.end('File not found: ' + filePath);
      } else {
        // Server error
        res.writeHead(500);
        res.end('Server Error: ' + error.code);
      }
      console.error('Error serving file:', error);
      return;
    }
    
    // Serve the file
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(content, 'utf-8');
  });
});

// Start the server
server.listen(PORT, () => {
  console.log(`Frontend server running at http://localhost:${PORT}`);
  console.log(`Serving HTML file from: ${path.join(__dirname, 'public', 'index.html')}`);
}); 