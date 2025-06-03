#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

console.log('🔧 Fixing all TypeScript compilation errors...\n');

// Fix TradeDetailsModal.tsx
console.log('📁 Fixing TradeDetailsModal.tsx...');
const tradeDetailsPath = './components/TradeDetailsModal.tsx';
if (fs.existsSync(tradeDetailsPath)) {
  let content = fs.readFileSync(tradeDetailsPath, 'utf8');
  
  // Fix trade.type vs trade.side comparison
  content = content.replace(/trade\.type === 'buy'/g, 'trade.side === \'buy\'');
  content = content.replace(/trade\.type === 'sell'/g, 'trade.side === \'sell\'');
  
  // Fix trade.status comparison (executed -> completed)
  content = content.replace(/trade\.status === 'executed'/g, 'trade.status === \'completed\'');
  
  // Fix trade.amount -> trade.quantity
  content = content.replace(/trade\.amount/g, 'trade.quantity');
  
  // Fix trade.value -> calculated value
  content = content.replace(/trade\.value/g, '(trade.price * trade.quantity)');
  
  // Fix trade.profit -> trade.pnl
  content = content.replace(/trade\.profit/g, 'trade.pnl');
  
  fs.writeFileSync(tradeDetailsPath, content, 'utf8');
  console.log('  ✅ Fixed TradeDetailsModal.tsx');
}

// Fix PerformanceMetrics.tsx
console.log('📁 Fixing PerformanceMetrics.tsx...');
const perfMetricsPath = './components/PerformanceMetrics.tsx';
if (fs.existsSync(perfMetricsPath)) {
  let content = fs.readFileSync(perfMetricsPath, 'utf8');
  
  // Fix import statement
  content = content.replace(/Cartesian Tooltip/g, 'CartesianGrid, Tooltip');
  
  // Remove Grid usage
  content = content.replace(/<Grid container[^>]*>/g, '<Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 3, mb: 4 }}>');
  content = content.replace(/<\/Grid>/g, '</Box>');
  
  fs.writeFileSync(perfMetricsPath, content, 'utf8');
  console.log('  ✅ Fixed PerformanceMetrics.tsx');
}

// Fix LightweightChart.tsx
console.log('📁 Fixing LightweightChart.tsx...');
const chartPath = './components/charts/LightweightChart.tsx';
if (fs.existsSync(chartPath)) {
  let content = fs.readFileSync(chartPath, 'utf8');
  
  // Fix null reference
  content = content.replace(
    /const chart = chartLib\.createChart\(chartContainerRef\.current!, \{\s*width: chartContainerRef\.current\.clientWidth,/,
    'const chartContainer = chartContainerRef.current!;\n      const chart = chartLib.createChart(chartContainer, {\n      width: chartContainer.clientWidth,'
  );
  
  // Fix param type
  content = content.replace(/chart\.subscribeCrosshairMove\(\(param\) =>/g, 'chart.subscribeCrosshairMove((param: any) =>');
  
  // Fix error type
  content = content.replace(/error: error\.message/g, 'error: error instanceof Error ? error.message : String(error)');
  
  fs.writeFileSync(chartPath, content, 'utf8');
  console.log('  ✅ Fixed LightweightChart.tsx');
}

// Fix MLIntelligenceDashboard.tsx
console.log('📁 Fixing MLIntelligenceDashboard.tsx...');
const mlDashPath = './components/intelligence/MLIntelligenceDashboard.tsx';
if (fs.existsSync(mlDashPath)) {
  let content = fs.readFileSync(mlDashPath, 'utf8');
  
  // Fix JSX > symbols
  content = content.replace(/Target: >1\.5/g, 'Target: &gt;1.5');
  content = content.replace(/Target: >1\.0/g, 'Target: &gt;1.0');
  
  fs.writeFileSync(mlDashPath, content, 'utf8');
  console.log('  ✅ Fixed MLIntelligenceDashboard.tsx');
}

// Fix NavigationSidebar.tsx
console.log('📁 Fixing NavigationSidebar.tsx...');
const navSidebarPath = './components/layout/NavigationSidebar.tsx';
if (fs.existsSync(navSidebarPath)) {
  let content = fs.readFileSync(navSidebarPath, 'utf8');
  
  // Fix pathname null check
  content = content.replace(
    /const isActive = \(href: string\) => \{\s*if \(!pathname\) return false;\s*if \(href === '\/dashboard'\) \{\s*return pathname === '\/' \|\| pathname === '\/dashboard';\s*\}\s*return pathname\.startsWith\(href\);\s*\};/,
    `const isActive = (href: string) => {
    if (!pathname) return false;
    if (href === '/dashboard') {
      return pathname === '/' || pathname === '/dashboard';
    }
    return pathname.startsWith(href);
  };`
  );
  
  fs.writeFileSync(navSidebarPath, content, 'utf8');
  console.log('  ✅ Fixed NavigationSidebar.tsx');
}

// Fix Layout.tsx
console.log('📁 Fixing Layout.tsx...');
const layoutPath = './components/layout/Layout.tsx';
if (fs.existsSync(layoutPath)) {
  let content = fs.readFileSync(layoutPath, 'utf8');
  
  // Fix import
  content = content.replace(/import Sidebar from '\.\/Sidebar';/, 'import { Sidebar } from \'./NavigationSidebar\';');
  
  // Remove props from Sidebar
  content = content.replace(
    /<Sidebar\s+open=\{[^}]+\}\s+onToggle=\{[^}]+\}\s+mobileOpen=\{[^}]+\}\s+onMobileClose=\{[^}]+\}\s*\/>/,
    '<Sidebar />'
  );
  
  fs.writeFileSync(layoutPath, content, 'utf8');
  console.log('  ✅ Fixed Layout.tsx');
}

// Fix layout/index.ts
console.log('📁 Fixing layout/index.ts...');
const layoutIndexPath = './components/layout/index.ts';
if (fs.existsSync(layoutIndexPath)) {
  let content = fs.readFileSync(layoutIndexPath, 'utf8');
  
  // Fix exports
  content = content.replace(/export \{ default as Sidebar \} from '\.\/Sidebar';/, '');
  content = content.replace(/export \{ default as NavigationSidebar \} from '\.\/NavigationSidebar';/, 'export { Sidebar as NavigationSidebar } from \'./NavigationSidebar\';');
  content = content.replace(/export \{ default as TopBar \} from '\.\/TopBar';/, 'export { TopBar } from \'./TopBar\';');
  
  fs.writeFileSync(layoutIndexPath, content, 'utf8');
  console.log('  ✅ Fixed layout/index.ts');
}

// Fix ThemeShowcase.tsx
console.log('📁 Fixing ThemeShowcase.tsx...');
const themeShowcasePath = './components/layout/ThemeShowcase.tsx';
if (fs.existsSync(themeShowcasePath)) {
  let content = fs.readFileSync(themeShowcasePath, 'utf8');
  
  // Fix import
  content = content.replace(/import \{ useTheme \} from '\.\.\/\.\.\/lib\/theme';/, 'import { lightTheme, darkTheme } from \'../../lib/theme\';');
  
  // Fix usage
  content = content.replace(
    /const \{ darkMode \} = useTheme\(\);\s*const theme = useMuiTheme\(\);/,
    'const theme = useMuiTheme();\n  const darkMode = theme.palette.mode === \'dark\';'
  );
  
  fs.writeFileSync(themeShowcasePath, content, 'utf8');
  console.log('  ✅ Fixed ThemeShowcase.tsx');
}

// Fix LayoutSettings.tsx
console.log('📁 Fixing LayoutSettings.tsx...');
const layoutSettingsPath = './components/settings/LayoutSettings.tsx';
if (fs.existsSync(layoutSettingsPath)) {
  let content = fs.readFileSync(layoutSettingsPath, 'utf8');
  
  // Fix ListItem button prop
  content = content.replace(
    /<ListItem\s+key=\{[^}]+\}\s+button\s+onClick=\{[^}]+\}>/g,
    '<ListItem\n                    key={widget.id}\n                    component="div"\n                    sx={{ cursor: \'pointer\' }}\n                    onClick={() => handleWidgetToggle(widget.id)}>'
  );
  
  fs.writeFileSync(layoutSettingsPath, content, 'utf8');
  console.log('  ✅ Fixed LayoutSettings.tsx');
}

// Fix EnhancedMonitoringDashboard.tsx
console.log('📁 Fixing EnhancedMonitoringDashboard.tsx...');
const monitoringPath = './components/monitoring/EnhancedMonitoringDashboard.tsx';
if (fs.existsSync(monitoringPath)) {
  let content = fs.readFileSync(monitoringPath, 'utf8');
  
  // Remove Grid usage
  content = content.replace(/<Grid container[^>]*>/g, '<Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(4, 1fr)" }, gap: 3, mb: 3 }}>');
  content = content.replace(/<\/Grid>/g, '</Box>');
  
  fs.writeFileSync(monitoringPath, content, 'utf8');
  console.log('  ✅ Fixed EnhancedMonitoringDashboard.tsx');
}

console.log('\n🏁 TypeScript error fixes complete!');
console.log('💡 Run "npm run build" to test the fixes');
