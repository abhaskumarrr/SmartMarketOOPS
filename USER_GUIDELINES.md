# 📋 SmartMarketOOPS User Guidelines & Development Workflow

**Version**: 2.0  
**Last Updated**: January 6, 2025  
**Status**: ✅ **ACTIVE** - Comprehensive development guidelines

---

## 🎯 **CORE DEVELOPMENT PRINCIPLES**

### **1. MCP-First Development Approach**
- **Always search for MCP servers** before implementing features manually
- **Use Context7 tool** for library documentation and best practices
- **Leverage web search** for latest solutions and troubleshooting
- **Integrate multiple MCP servers** for comprehensive development support

### **2. Quality-First Implementation**
- **Professional-grade code**: Every component must meet production standards
- **Performance optimization**: MacBook Air M2 (8GB RAM) friendly
- **Comprehensive testing**: Unit, integration, and E2E tests required
- **Documentation**: All features must be documented and have examples

### **3. Systematic Problem-Solving**
- **7-Step Approach**: Examine → Review → Identify → Plan → Confirm → Implement → Update
- **Break down complexity**: Large features into manageable chunks
- **3-Phase Implementation**: Infrastructure → Enhancement → Production Readiness
- **Validation after each phase**: Ensure stability before proceeding

---

## 🛠 **RECOMMENDED MCP SERVERS**

### **Essential Development MCP Servers**
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["@context7/mcp-server"],
      "description": "Library documentation and best practices"
    },
    "github-integration": {
      "command": "npx", 
      "args": ["@github/mcp-server"],
      "env": { "GITHUB_TOKEN": "your_token" }
    },
    "web-search": {
      "command": "npx",
      "args": ["@brave/search-mcp-server"],
      "env": { "BRAVE_API_KEY": "your_key" }
    },
    "file-operations": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem"],
      "args": ["/Users/abhaskumarrr/Documents/GitHub/SmartMarketOOPS"]
    }
  }
}
```

### **Frontend-Specific MCP Servers**
```json
{
  "mcpServers": {
    "figma-to-react": {
      "command": "npx",
      "args": ["@figma/mcp-server"],
      "description": "Convert Figma designs to React components"
    },
    "storybook-integration": {
      "command": "npx", 
      "args": ["@storybook/mcp-server"],
      "description": "Component documentation and testing"
    },
    "performance-monitor": {
      "command": "npx",
      "args": ["@lighthouse/mcp-server"],
      "description": "Performance monitoring and optimization"
    },
    "accessibility-audit": {
      "command": "npx",
      "args": ["@axe-core/mcp-server"],
      "description": "Automated accessibility testing"
    }
  }
}
```

### **Trading-Specific MCP Servers**
```json
{
  "mcpServers": {
    "delta-exchange": {
      "command": "npx",
      "args": ["@delta-exchange/mcp-server"],
      "description": "Delta Exchange API integration"
    },
    "trading-analytics": {
      "command": "npx",
      "args": ["@trading/analytics-mcp-server"],
      "description": "Trading performance analytics"
    },
    "ml-models": {
      "command": "npx",
      "args": ["@ml/trading-models-mcp-server"],
      "description": "ML model integration and monitoring"
    }
  }
}
```

---

## 🏗 **FRONTEND DEVELOPMENT WORKFLOW**

### **Phase 1: Foundation & Cleanup**
1. **Component Audit**: Remove duplicates and inconsistencies
2. **Dependency Management**: Standardize on Shadcn/UI + Tailwind
3. **File Structure**: Organize components by feature and responsibility
4. **Design System**: Implement consistent component patterns

### **Phase 2: Feature Implementation**
1. **Real-time Dashboard**: Live trading data and portfolio monitoring
2. **Bot Management**: Comprehensive bot creation and monitoring
3. **Analytics Interface**: Advanced trading analytics and insights
4. **User Experience**: Responsive design and smooth animations

### **Phase 3: Production Readiness**
1. **Performance Optimization**: Bundle analysis and lazy loading
2. **Testing Coverage**: Unit, integration, and E2E tests
3. **Error Handling**: Comprehensive error boundaries and recovery
4. **Documentation**: Complete API docs and user guides

---

## 📊 **TRADING SYSTEM GUIDELINES**

### **Performance Requirements**
- **Win Rate**: Target 60%+ (Current: 65%+)
- **Sharpe Ratio**: Target 1.5+ (Current: 1.5+)
- **Max Drawdown**: Limit 20% (Current: <5%)
- **Daily Signals**: Generate 3-5 high-confidence trades
- **Leverage**: 100x BTC/ETH, 50x SOL, 75% balance allocation

### **Risk Management Rules**
- **Dynamic Position Sizing**: Adjust based on confidence and volatility
- **Multi-timeframe Analysis**: 5m, 15m, 1h, 4h hierarchy
- **Fibonacci Retracement**: Monthly timeframe for swing analysis
- **ML-Driven Decisions**: Use AI models as primary decision makers
- **Real-time Monitoring**: Continuous market analysis and adaptation

### **Delta Exchange Configuration**
- **API Credentials**: Configured in .env file
- **Product IDs**: BTC (27), ETH (3136) perpetual futures
- **Testnet First**: Always test on testnet before production
- **Position Limits**: Maximum 2 active positions
- **Balance Management**: Stop trading when insufficient balance

---

## 🔧 **DEVELOPMENT STANDARDS**

### **Code Quality Requirements**
```typescript
// Example: Proper TypeScript component structure
interface TradingComponentProps {
  symbol: string;
  timeframe: '5m' | '15m' | '1h' | '4h';
  onSignalGenerated: (signal: TradingSignal) => void;
}

export const TradingComponent: React.FC<TradingComponentProps> = ({
  symbol,
  timeframe,
  onSignalGenerated
}) => {
  // Implementation with proper error handling
  // Performance optimization with React.memo
  // Accessibility compliance
  // Comprehensive testing
};
```

### **Testing Requirements**
- **Unit Tests**: All business logic components
- **Integration Tests**: API interactions and data flow
- **E2E Tests**: Critical user journeys
- **Performance Tests**: Bundle size and load times
- **Accessibility Tests**: WCAG 2.1 AA compliance

### **Documentation Standards**
- **Component Stories**: Storybook for all UI components
- **API Documentation**: OpenAPI/Swagger for all endpoints
- **User Guides**: Step-by-step usage instructions
- **Technical Docs**: Architecture and deployment guides

---

## 🚀 **IMPLEMENTATION CHECKLIST**

### **Before Starting Any Feature**
- [ ] Search for relevant MCP servers
- [ ] Use Context7 for library documentation
- [ ] Web search for latest best practices
- [ ] Create detailed implementation plan
- [ ] Get user confirmation before proceeding

### **During Development**
- [ ] Follow established patterns and conventions
- [ ] Write tests alongside implementation
- [ ] Use MCP servers for assistance and validation
- [ ] Monitor performance and bundle size
- [ ] Ensure accessibility compliance

### **After Implementation**
- [ ] Update TaskMaster status
- [ ] Create/update documentation
- [ ] Run comprehensive tests
- [ ] Performance validation
- [ ] User acceptance testing

---

## 📈 **SUCCESS METRICS**

### **Technical Metrics**
- **Performance**: Lighthouse score > 90
- **Bundle Size**: < 500KB gzipped
- **Test Coverage**: > 80%
- **Error Rate**: < 1%

### **Trading Metrics**
- **Win Rate**: > 60%
- **Profit Factor**: > 1.5
- **Maximum Drawdown**: < 20%
- **Signal Accuracy**: > 85%

### **User Experience Metrics**
- **Load Time**: < 2 seconds
- **Mobile Compatibility**: 100%
- **Accessibility Score**: WCAG 2.1 AA
- **User Satisfaction**: > 90%

---

## 🔄 **CONTINUOUS IMPROVEMENT**

### **Regular Reviews**
- **Weekly**: Code quality and performance review
- **Monthly**: Trading performance analysis
- **Quarterly**: Technology stack evaluation
- **Annually**: Complete system architecture review

### **Feedback Integration**
- **User Feedback**: Continuous collection and integration
- **Performance Monitoring**: Real-time system health tracking
- **Market Adaptation**: Regular strategy optimization
- **Technology Updates**: Stay current with latest developments

---

**Status**: 🟢 **ACTIVE GUIDELINES** - Ready for implementation

*These guidelines ensure SmartMarketOOPS maintains professional quality while leveraging the latest MCP server capabilities for enhanced development productivity.*
