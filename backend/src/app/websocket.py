from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"New client connected: {websocket.client}. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected: {websocket.client}. Total clients: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Message received from {client_id}: {data}")
            await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")


    async def broadcast_market_data(self):
        logger.info("Connecting to Delta Exchange websocket...")
        logger.info("Connecting to Delta Exchange websocket...")
        uri = "wss://testnet-socket.delta.exchange"
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to Delta Exchange websocket.")
            logger.info("Connected to Delta Exchange websocket.")
            subscription_message = {
                "type": "subscribe",
                "payload": {
                    "channels": [
                        {
                            "name": "v2/ticker",
                            "symbols": self.symbols
                        }
                    ]
                }
            }
            await websocket.send(json.dumps(subscription_message))
            while True:
                try:
                    message = await websocket.recv()
                    logger.info(f"Received message from Delta Exchange: {message}")
                    logger.info(f"Received message from Delta Exchange: {message}")
                    data = json.loads(message)
                    if data.get("type") == "v2/ticker":
                        self.last_market_data[data["symbol"]] = data
                        for client in self.clients:
                            await client.send_text(json.dumps({"type": "market_data", "data": data}))
                except ConnectionClosed:
                    break

