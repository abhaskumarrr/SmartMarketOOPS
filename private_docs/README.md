# SmartMarketOOPS - Private Project Documentation

## 1. Project Overview

This document provides a comprehensive overview of the SmartMarketOOPS project, intended for private use.

SmartMarketOOPS is a production-grade, fully automated machine learning pipeline for cryptocurrency trading that implements institutional-level Smart Money Concepts (SMC), advanced technical analysis, and multi-timeframe confluence strategies. The platform combines sophisticated ML models with real-time market analysis to generate high-probability trading signals and execute trades automatically.

- **Backend**: Node.js, Express, PostgreSQL, Prisma, WebSocket
- **Frontend**: Next.js, React 19, TypeScript, Tailwind CSS, Recharts
- **Machine Learning**: Python, PyTorch, FastAPI

---

## 2. File Structure

Below is a snapshot of the key directories and their purposes.

```
SmartMarketOOPS/
├── backend/                  # Node.js backend services, API, and database logic
│   ├── prisma/               # Prisma schema and migration files
│   └── src/                  # Backend source code
├── frontend/                 # Next.js frontend application
│   ├── public/               # Static assets
│   └── src/                  # Frontend source code
│       ├── app/              # Next.js App Router pages and layouts
│       ├── components/       # Reusable React components
│       ├── hooks/            # Custom React hooks
│       └── lib/              # Utility functions and libraries
├── ml/                       # Python machine learning services
│   ├── models/               # Trained ML models
│   └── src/                  # ML source code
├── private_docs/             # Private project documentation (this folder)
├── scripts/                  # General-purpose scripts for setup and automation
├── .gitignore                # Specifies intentionally untracked files to ignore
├── docker-compose.yml        # Defines and runs multi-container Docker applications
├── package.json              # Defines project metadata and dependencies (root)
└── README.md                 # Public-facing README file
```

---

## 3. Core Dependencies

### 3.1. Root Dependencies (`package.json`)

These packages manage the overall monorepo, workspaces, and run concurrent scripts.

- **`concurrently`**: Runs multiple commands concurrently (e.g., backend and frontend servers).
- **`dotenv`**: Loads environment variables from a `.env` file.
- **`eslint`**: Linter for identifying and reporting on patterns in JavaScript.
- **`typescript`**: Superset of JavaScript that adds static types.
- **`ccxt`**: A JavaScript / Python / PHP cryptocurrency trading API with support for many exchanges.
- **`next`**: The React framework for production.
- **`task-master-ai`**: AI-powered task management tool.

### 3.2. Frontend Dependencies (`frontend/package.json`)

These packages are specific to the Next.js user interface.

- **`@radix-ui/*`**: A collection of unstyled, accessible UI components.
- **`autoprefixer`**: A PostCSS plugin to parse CSS and add vendor prefixes.
- **`class-variance-authority`**: Create responsive, type-safe UI components.
- **`clsx`**: A tiny utility for constructing `className` strings conditionally.
- **`framer-motion`**: A production-ready motion library for React.
- **`lightweight-charts`**: Financial charting library.
- **`lucide-react`**: A simple and beautiful icon library.
- **`next`**: The React framework for the frontend.
- **`next-themes`**: An abstraction for themes in Next.js.
- **`react` / `react-dom`**: Core libraries for building user interfaces.
- **`recharts`**: A composable charting library built on React components.
- **`socket.io-client`**: Real-time engine for WebSocket communication.
- **`tailwind-merge`**: Utility function to merge Tailwind CSS classes.
- **`tailwindcss`**: A utility-first CSS framework.
- **`tailwindcss-animate`**: A Tailwind CSS plugin for creating animations.
- **`zustand`**: A small, fast, and scalable state-management solution.

### 3.3. Python Dependencies (`requirements.txt`)

Key packages for the machine learning service.

- **`fastapi`**: A modern, fast web framework for building APIs with Python.
- **`torch`**: An open source machine learning framework that accelerates the path from research to production.
- **`pandas`**: A fast, powerful, flexible and easy to use open source data analysis and manipulation tool.
- **`scikit-learn`**: Simple and efficient tools for predictive data analysis.
- **`uvicorn`**: An ASGI server implementation, for running FastAPI applications.
- **`python-socketio`**: Python implementation of the Socket.IO realtime server.

---

This document should serve as a starting point. Refer to the other files in this directory for more specific details on PRD, completed tasks, and known issues. 