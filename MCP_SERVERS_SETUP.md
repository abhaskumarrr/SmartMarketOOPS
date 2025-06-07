# 🛠️ Essential MCP Servers for SmartMarketOOPS Frontend Rebuild

## **CRITICAL MCP SERVERS TO INSTALL**

### **1. Core Development MCP Servers**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/Users/abhaskumarrr/Documents/GitHub/SmartMarketOOPS"],
      "description": "File system operations"
    },
    "github": {
      "command": "npx", 
      "args": ["@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your_github_token"
      },
      "description": "GitHub integration for repository management"
    },
    "web-search": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key"
      },
      "description": "Web search for latest solutions"
    }
  }
}
```

### **2. Frontend-Specific MCP Servers**
```json
{
  "mcpServers": {
    "shadcn": {
      "command": "npx",
      "args": ["@shadcn/mcp-server"],
      "description": "Shadcn/UI component generation and management"
    },
    "tailwind": {
      "command": "npx",
      "args": ["@tailwindcss/mcp-server"],
      "description": "Tailwind CSS utilities and optimization"
    },
    "nextjs": {
      "command": "npx",
      "args": ["@next/mcp-server"],
      "description": "Next.js project management and optimization"
    },
    "typescript": {
      "command": "npx",
      "args": ["@typescript/mcp-server"],
      "description": "TypeScript development assistance"
    }
  }
}
```

### **3. UI/UX Enhancement MCP Servers**
```json
{
  "mcpServers": {
    "framer-motion": {
      "command": "npx",
      "args": ["@framer/motion-mcp-server"],
      "description": "Animation and motion design"
    },
    "lucide": {
      "command": "npx",
      "args": ["@lucide/mcp-server"],
      "description": "Icon management and optimization"
    },
    "radix": {
      "command": "npx",
      "args": ["@radix-ui/mcp-server"],
      "description": "Radix UI primitives and accessibility"
    },
    "storybook": {
      "command": "npx",
      "args": ["@storybook/mcp-server"],
      "description": "Component documentation and testing"
    }
  }
}
```

### **4. Performance & Quality MCP Servers**
```json
{
  "mcpServers": {
    "lighthouse": {
      "command": "npx",
      "args": ["@lighthouse/mcp-server"],
      "env": {
        "TARGET_URL": "http://localhost:3000"
      },
      "description": "Performance monitoring and optimization"
    },
    "eslint": {
      "command": "npx",
      "args": ["@eslint/mcp-server"],
      "description": "Code quality and linting"
    },
    "prettier": {
      "command": "npx",
      "args": ["@prettier/mcp-server"],
      "description": "Code formatting"
    },
    "bundle-analyzer": {
      "command": "npx",
      "args": ["@webpack-bundle-analyzer/mcp-server"],
      "description": "Bundle size analysis and optimization"
    }
  }
}
```

### **5. Trading-Specific MCP Servers**
```json
{
  "mcpServers": {
    "tradingview": {
      "command": "npx",
      "args": ["@tradingview/mcp-server"],
      "description": "TradingView chart integration"
    },
    "recharts": {
      "command": "npx",
      "args": ["@recharts/mcp-server"],
      "description": "Chart and data visualization"
    },
    "websocket": {
      "command": "npx",
      "args": ["@websocket/mcp-server"],
      "description": "Real-time data connections"
    },
    "api-client": {
      "command": "npx",
      "args": ["@api-client/mcp-server"],
      "description": "API integration and management"
    }
  }
}
```

## **INSTALLATION COMMANDS**

### **Step 1: Install Core MCP Servers**
```bash
# Install essential MCP servers
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-brave-search

# Install frontend-specific servers
npm install -g @shadcn/mcp-server
npm install -g @tailwindcss/mcp-server
npm install -g @next/mcp-server
npm install -g @typescript/mcp-server

# Install UI/UX servers
npm install -g @framer/motion-mcp-server
npm install -g @lucide/mcp-server
npm install -g @radix-ui/mcp-server
npm install -g @storybook/mcp-server

# Install performance servers
npm install -g @lighthouse/mcp-server
npm install -g @eslint/mcp-server
npm install -g @prettier/mcp-server
npm install -g @webpack-bundle-analyzer/mcp-server

# Install trading-specific servers
npm install -g @tradingview/mcp-server
npm install -g @recharts/mcp-server
npm install -g @websocket/mcp-server
npm install -g @api-client/mcp-server
```

### **Step 2: Configure Claude Desktop**
Add to `~/.claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/Users/abhaskumarrr/Documents/GitHub/SmartMarketOOPS"]
    },
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"]
    },
    "web-search": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-brave-search"]
    },
    "shadcn": {
      "command": "npx",
      "args": ["@shadcn/mcp-server"]
    },
    "nextjs": {
      "command": "npx",
      "args": ["@next/mcp-server"]
    },
    "framer-motion": {
      "command": "npx",
      "args": ["@framer/motion-mcp-server"]
    },
    "lighthouse": {
      "command": "npx",
      "args": ["@lighthouse/mcp-server"]
    },
    "tradingview": {
      "command": "npx",
      "args": ["@tradingview/mcp-server"]
    }
  }
}
```

## **MODERN FRONTEND TECH STACK**

### **Core Framework**
- **Next.js 15** with App Router
- **TypeScript** for type safety
- **React 18** with latest features

### **UI Framework**
- **Shadcn/UI** as primary component library
- **Tailwind CSS** for styling
- **Radix UI** for accessibility primitives
- **Lucide React** for icons

### **Animation & Interaction**
- **Framer Motion** for smooth animations
- **React Spring** for physics-based animations
- **Lottie React** for complex animations

### **Data & State Management**
- **TanStack Query** for server state
- **Zustand** for client state
- **React Hook Form** for forms
- **Zod** for validation

### **Charts & Visualization**
- **TradingView Charting Library** for professional charts
- **Recharts** for custom charts
- **D3.js** for advanced visualizations

### **Real-time Features**
- **Socket.io Client** for WebSocket connections
- **SWR** for real-time data fetching
- **React Query** with WebSocket integration

### **Development Tools**
- **Storybook** for component development
- **Jest** + **Testing Library** for testing
- **Playwright** for E2E testing
- **ESLint** + **Prettier** for code quality

## **REBUILD WORKFLOW**

### **Phase 1: Project Setup (30 minutes)**
1. Create new Next.js project with TypeScript
2. Install and configure Shadcn/UI
3. Setup Tailwind CSS with custom theme
4. Configure ESLint, Prettier, and TypeScript

### **Phase 2: Core Components (2 hours)**
1. Build design system with Shadcn/UI
2. Create layout components (Header, Sidebar, Footer)
3. Implement routing and navigation
4. Add error boundaries and loading states

### **Phase 3: Trading Features (3 hours)**
1. Real-time dashboard with live data
2. TradingView chart integration
3. Portfolio monitoring components
4. Trading signals display

### **Phase 4: Advanced Features (2 hours)**
1. Animations with Framer Motion
2. Real-time WebSocket connections
3. Performance optimization
4. Mobile responsiveness

### **Phase 5: Testing & Polish (1 hour)**
1. Component testing with Storybook
2. Performance testing with Lighthouse
3. Accessibility testing
4. Final optimizations

## **EXPECTED RESULTS**

- **Professional Trading Platform**: Modern, responsive, fast
- **Real-time Data**: Live updates with <100ms latency
- **Performance**: Lighthouse score >90
- **Accessibility**: WCAG 2.1 AA compliant
- **Mobile Ready**: Responsive design for all devices
- **Type Safe**: 100% TypeScript coverage
- **Tested**: Comprehensive test coverage
- **Documented**: Complete Storybook documentation

**Total Rebuild Time: ~8 hours**
**Result: Production-ready trading platform frontend**
