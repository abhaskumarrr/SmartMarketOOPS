import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8002/ws/1"
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        
        # Send a message
        await websocket.send("Hello, WebSocket!")
        print("> Hello, WebSocket!")

        # Receive a message
        response = await websocket.recv()
        print(f"< {response}")

if __name__ == "__main__":
    asyncio.run(test_websocket())

