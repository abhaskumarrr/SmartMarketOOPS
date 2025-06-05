# SmartMarketOOPS

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/abhaskumarrr/SmartMarketOOPS/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **AI-powered, real-time crypto trading platform with ML, SMC, and institutional-grade analytics.**

---

## 🚀 Project Overview
SmartMarketOOPS is a full-stack, production-grade trading system for cryptocurrency markets. It combines advanced machine learning (ML), Smart Money Concepts (SMC), and real-time dashboards to deliver institutional-level trading automation, analytics, and portfolio management.

---

## ✨ Key Features
- Real-time trading bot with live market data (Binance, Delta Exchange)
- ML-driven signal generation (LSTM, CNN, Transformer, Ensemble)
- SMC modules: Order Block, FVG, Liquidity, Market Structure, Multi-Timeframe Confluence
- Secure API key management and user authentication (JWT, RBAC)
- Responsive dashboard with TradingView charts, performance metrics, and user settings
- Free-tier infrastructure: Vercel, Railway, Supabase, Hugging Face, GitHub Actions
- Comprehensive test suite and CI/CD pipeline

---

## ⚡ Quick Start

### 1. **Clone the Repo**
```bash
git clone https://github.com/abhaskumarrr/SmartMarketOOPS.git
cd SmartMarketOOPS
```

### 2. **Setup Environment**
- Copy `.env` files from `example.env` and fill in your secrets.
- Install dependencies:
```bash
  npm install
  cd backend && npm install
  cd ../frontend && npm install
  cd ..
  ```

### 3. **Start Infrastructure (Docker Compose)**
```bash
docker compose -f docker-compose.infrastructure.yml up -d
```

### 4. **Run Backend & Frontend**
```bash
# In one terminal
cd backend && npm run dev
# In another terminal
cd frontend && npm run dev
```

---

## 🖥️ Usage
- **Trading Bot:**
```bash
  cd backend
  npm run trade:working-system
  ```
- **Dashboard:**
  - Visit [http://localhost:3000](http://localhost:3000) after starting the frontend.
- **API Docs:**
  - See `backend/docs/` for OpenAPI/Swagger documentation.

---

## 🏗️ Architecture

```mermaid
graph TD;
  A[Frontend (Next.js)] -->|REST/WebSocket| B[Backend (Node.js/Express)]
  B -->|ML API| C[ML Service (Python/FastAPI)]
  B -->|PostgreSQL| D[(Database)]
  B -->|QuestDB| E[(Time-Series DB)]
  B -->|Redis| F[(Event Streams)]
  B -->|SMC/ML| G[Trading Engine]
  G -->|Exchange API| H[Binance/Delta Exchange]
```

---

## ⚙️ Configuration
- All secrets and API keys are managed via `.env` files (see `example.env`).
- Docker Compose manages infrastructure services (Postgres, Redis, QuestDB).
- See `SMARTMARKETOOPS_LAUNCH_GUIDE.md` for full setup.

---

## 🧪 Testing
- **Backend:**
  ```bash
   cd backend
  npm run test
  ```
- **Frontend:**
  ```bash
   cd frontend
  npm run test
  ```
- **ML:**
  ```bash
  cd ml
  pytest
  ```

---

## 🚀 Deployment
- **Frontend:** Vercel
- **Backend:** Railway
- **ML Service:** Hugging Face Spaces
- **Database:** Supabase (Postgres), QuestDB (Time-Series)
- **CI/CD:** GitHub Actions

---

## 🤝 Contributing
1. Fork the repo and create a feature branch.
2. Make your changes and add tests.
3. Submit a pull request with a clear description.
4. See `CONTRIBUTING.md` for more details.

---

## 🛠️ Troubleshooting & FAQ
- **Docker Compose warning about `version`:** Remove the `version` key from your compose file ([details](https://adamj.eu/tech/2025/05/05/docker-remove-obsolete-compose-version/)).
- **Trade failed warnings:** Check trade execution logic and logs for order size, price, or API issues.
- **Environment issues:** Ensure all `.env` variables are set and Docker services are running.

---

## 📄 License
MIT. See [LICENSE](LICENSE).

---

##  Acknowledgements
- [Cursor](https://cursor.com/) for AI agent workflow
- [Task Master](https://forum.cursor.com/t/task-master-prompt-agent-mode/39980) for project/task management
- [BuzzFeed Taskmaster](https://www.buzzfeed.com/hopelasater/best-taskmaster-tasks-ranked) for inspiration
- All open source contributors and the crypto/ML community