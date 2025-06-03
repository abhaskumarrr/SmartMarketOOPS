#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Find all TypeScript/JavaScript files in components directory
function findFiles(dir, extension = '.tsx') {
  let results = [];
  const list = fs.readdirSync(dir);
  
  list.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat && stat.isDirectory()) {
      results = results.concat(findFiles(filePath, extension));
    } else if (file.endsWith(extension) || file.endsWith('.ts')) {
      results.push(filePath);
    }
  });
  
  return results;
}

// Fix Grid components in a file
function fixGridComponents(filePath) {
  console.log(`Processing: ${filePath}`);
  
  let content = fs.readFileSync(filePath, 'utf8');
  let modified = false;
  
  // Remove Grid from imports if it exists
  const gridImportRegex = /,\s*Grid,/g;
  if (content.match(gridImportRegex)) {
    content = content.replace(gridImportRegex, ',');
    modified = true;
    console.log(`  ✓ Removed Grid import`);
  }
  
  // Replace Grid container with Box
  const gridContainerRegex = /<Grid\s+container\s+spacing=\{(\d+)\}>/g;
  const gridContainerMatches = content.match(gridContainerRegex);
  if (gridContainerMatches) {
    content = content.replace(gridContainerRegex, (match, spacing) => {
      return `<Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' }, gap: ${spacing} }}>`;
    });
    modified = true;
    console.log(`  ✓ Replaced Grid container with Box`);
  }
  
  // Replace Grid items with Box
  const gridItemRegex = /<Grid\s+item\s+xs=\{(\d+)\}(?:\s+sm=\{(\d+)\})?(?:\s+md=\{(\d+)\})?>/g;
  const gridItemMatches = content.match(gridItemRegex);
  if (gridItemMatches) {
    content = content.replace(gridItemRegex, '<Box>');
    modified = true;
    console.log(`  ✓ Replaced Grid items with Box`);
  }
  
  // Replace closing Grid tags
  const closingGridRegex = /<\/Grid>/g;
  const closingGridMatches = content.match(closingGridRegex);
  if (closingGridMatches) {
    content = content.replace(closingGridRegex, '</Box>');
    modified = true;
    console.log(`  ✓ Replaced closing Grid tags with Box`);
  }
  
  // Handle more complex Grid patterns
  const complexGridRegex = /<Grid\s+[^>]*item[^>]*>/g;
  const complexGridMatches = content.match(complexGridRegex);
  if (complexGridMatches) {
    content = content.replace(complexGridRegex, '<Box>');
    modified = true;
    console.log(`  ✓ Replaced complex Grid items with Box`);
  }
  
  if (modified) {
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`  ✅ File updated successfully`);
  } else {
    console.log(`  ⏭️  No Grid components found`);
  }
  
  return modified;
}

// Main execution
function main() {
  console.log('🔧 Fixing Material-UI Grid components...\n');
  
  const componentsDir = './components';
  const appDir = './app';
  
  let totalModified = 0;
  
  // Process components directory
  if (fs.existsSync(componentsDir)) {
    console.log('📁 Processing components directory...');
    const componentFiles = findFiles(componentsDir);
    
    componentFiles.forEach(file => {
      if (fixGridComponents(file)) {
        totalModified++;
      }
    });
  }
  
  // Process app directory
  if (fs.existsSync(appDir)) {
    console.log('\n📁 Processing app directory...');
    const appFiles = findFiles(appDir);
    
    appFiles.forEach(file => {
      if (fixGridComponents(file)) {
        totalModified++;
      }
    });
  }
  
  console.log(`\n🏁 Grid component fix complete!`);
  console.log(`📊 Files modified: ${totalModified}`);
  
  if (totalModified > 0) {
    console.log('\n💡 Next steps:');
    console.log('1. Run "npm run build" to test the fixes');
    console.log('2. Review the changes to ensure layout is preserved');
    console.log('3. Test the UI to verify functionality');
  }
}

main();
