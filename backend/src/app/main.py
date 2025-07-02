import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

try:
    from fastapi import FastAPI, Depends
    from fastapi.responses import HTMLResponse
    from starlette.middleware.cors import CORSMiddleware
    import os
    from sqlalchemy.orm import Session

    # Import routers
    from .auth import router as auth_router
    # from .core_strategy_service import router as core_strategy_router
    # from .model_service import router as model_router
    from .websocket import router as websocket_router
    from .database import SessionLocal, engine, Base

    logging.info("Creating database tables...")
    # Create all tables
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables created.")

    logging.info("Initializing FastAPI app...")
    # --- FastAPI App Initialization ---

    app = FastAPI(
        title="SmartMarketOOPS - Enhanced Trading API",
        description="Provides high-probability trading signals and enhanced ML predictions.",
        version="2.0.0"
    )

    # Dependency
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # --- Middleware ---

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- API Routers ---

    app.include_router(auth_router)
    # app.include_router(core_strategy_router, tags=["strategy"])
    # app.include_router(model_router, tags=["models"])
    app.include_router(websocket_router, tags=["websockets"])


    # --- Root Endpoint ---

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return """
        <html>
            <head>
                <title>SmartMarketOOPS API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #333; }
                    p { color: #555; }
                    ul { list-style-type: none; padding: 0; }
                    li { background: #f4f4f4; margin: 5px 0; padding: 10px; border-radius: 5px; }
                    code { background: #e4e4e4; padding: 2px 6px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <h1>SmartMarketOOPS - Enhanced Trading API</h1>
                <p><strong>Status:</strong> <span style="color: green;">ok</span></p>
                <p>The API is running successfully. Here are the available endpoint categories:</p>
                <ul>
                    <li><code>/auth</code> - User authentication and token management.</li>
                    <li><code>/strategy</code> - Core trading strategy and signal generation.</li>
                    <li><code>/models</code> - Machine learning model predictions and management.</li>
                    <li><code>/health</code> - Service health check.</li>
                    <li><code>/docs</code> - OpenAPI (Swagger) documentation.</li>
                    <li><code>/redoc</code> - ReDoc documentation.</li>
                </ul>
            </body>
        </html>
        """

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "SmartMarketOOPS API",
            "timestamp": "2025-01-27T16:30:00Z"
        }

except Exception as e:
    logging.exception(f"An unexpected error occurred: {e}")
    sys.exit(1)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
