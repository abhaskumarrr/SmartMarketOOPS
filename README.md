# SmartMarketOOPS: A Human-Like Trading Agent

Welcome to SmartMarketOOPS, a sophisticated, AI-driven trading platform designed to learn, adapt, and trade like a seasoned expert. This project integrates multiple AI methodologies to create a trading agent that evolves over time, combining the best of human intuition and machine precision.

## Core Philosophy

The goal of this project is to move beyond simple, rule-based trading bots. We are building a **human-like trading agent** that learns through a multi-phased approach:
1.  **Imitation Learning:** The agent starts by learning from expert human traders, mimicking their strategies and decision-making processes.
2.  **Reinforcement Learning with Human Feedback (RLHF):** It then refines its understanding through simulated trading, where its actions are guided and corrected by human feedback.
3.  **Meta-Learning:** Finally, it learns *how to learn*, adapting its strategies to new and changing market conditions on its own.

For a detailed breakdown of the development plan, see [gemini-plan.md](gemini-plan.md).

## Getting Started

This project is a multi-component system. The following steps will get you up and running with a single command.

### Prerequisites
- Node.js >= 20.11.0
- Python >= 3.10
- `pnpm` for package management

### 1. Installation
First, clone the repository and install the dependencies for all workspaces:
```bash
git clone https://github.com/abhaskumarrr/SmartMarketOOPS.git
cd SmartMarketOOPS
pnpm install
```

### 2. Environment Setup
Copy the example environment file and update it with your credentials (e.g., your Delta Exchange API keys).
```bash
cp example.env .env
```
**Note:** The `.env` file is ignored by Git and should never be committed.

### 3. Launch the System
Use the unified startup script to launch the frontend, backend, and ML services.
```bash
./scripts/start_all.sh
```
Once all services are running, you can access the main dashboard at **http://localhost:3000**.

## Project Structure

The project is organized into three main components in a monorepo structure:

- **`frontend/`**: A Next.js and TypeScript application that provides the user interface for the trading dashboard.
- **`backend/`**: A Node.js and Express server that acts as the central orchestrator, handling API requests, exchange integrations, and communication between the other services.
- **`ml/`**: A Python-based service containing all the machine learning logic, including model training, signal generation, and the RL environment.

For a more detailed breakdown of the architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development Roadmap

The development of the human-like trading agent is divided into the following phases:

-   **Phase 0: Project Cleanup and Structuring** `(Completed)`
-   **Phase 1: Formalize Data Collection** `(Completed)`
-   **Phase 2: The Imitation Phase** `(Completed)`
-   **Phase 3: Reinforcement Learning with Human Feedback** `(In Progress)`
-   **Phase 4 & 5: Meta-Learning and Hybrid Deployment**

You can follow the detailed plan and progress in [gemini-plan.md](gemini-plan.md).

## Contributing

This project is under active development. Contributions are welcome. Please follow the standard Git workflow of forking the repository, creating a feature branch, and submitting a pull request.
