import asyncio
import pytest
from websockets.asyncio.server import serve
from beni_brain.relay import RelayBridge
from beni_common.schemas import pack, unpack


@pytest.mark.asyncio
async def test_relay_url_normalization():
    b1 = RelayBridge("https://relay.example.com", "tok", 8765)
    assert b1.relay_url == "wss://relay.example.com"

    b2 = RelayBridge("http://localhost:8080/", "tok", 8765)
    assert b2.relay_url == "ws://localhost:8080"

    b3 = RelayBridge("wss://relay.example.com/sub", "tok", 8765)
    assert b3.relay_url == "wss://relay.example.com/sub"


@pytest.mark.asyncio
async def test_relay_realtime_bridge_e2e():
    """Simulates a mock Cloudflare relay, mock local gateway, and verifies bidirectional frame flow."""
    relay_frames_rx = []
    local_frames_rx = []

    # 1. Mock Local Gateway (:port/ws)
    async def gateway_handler(ws):
        try:
            async for raw in ws:
                f = unpack(raw)
                local_frames_rx.append(f)
                if f.get("type") == "hello":
                    await ws.send(pack({"type": "hello_ok", "seq": 0, "t": 1.0}))
        except Exception:
            pass

    # 2. Mock Cloudflare Relay (:port/brain)
    relay_ws_holder = []
    async def relay_handler(ws):
        relay_ws_holder.append(ws)
        try:
            async for raw in ws:
                f = unpack(raw)
                relay_frames_rx.append(f)
        except Exception:
            pass

    async with serve(gateway_handler, "127.0.0.1", 0) as gw_server:
        gw_port = gw_server.sockets[0].getsockname()[1]

        async with serve(relay_handler, "127.0.0.1", 0) as relay_server:
            relay_port = relay_server.sockets[0].getsockname()[1]
            relay_url = f"ws://127.0.0.1:{relay_port}"

            bridge = RelayBridge(relay_url, token="test_token", local_port=gw_port)
            bridge_task = asyncio.create_task(bridge.run())

            try:
                # Wait for bridge to connect to mock relay
                for _ in range(50):
                    if relay_ws_holder:
                        break
                    await asyncio.sleep(0.05)
                assert len(relay_ws_holder) > 0, "Bridge never connected to mock relay"
                ws_to_brain = relay_ws_holder[0]

                # Simulate robot sending 'hello' to relay
                await ws_to_brain.send(pack({"type": "hello", "robot_id": "test-bot", "token": "test_token"}))

                # Verify local gateway received the hello
                for _ in range(50):
                    if len(local_frames_rx) > 0:
                        break
                    await asyncio.sleep(0.05)
                assert len(local_frames_rx) == 1
                assert local_frames_rx[0]["type"] == "hello"

                # Verify relay received the 'hello_ok' response from local gateway
                for _ in range(50):
                    if len(relay_frames_rx) > 0:
                        break
                    await asyncio.sleep(0.05)
                assert len(relay_frames_rx) == 1
                assert relay_frames_rx[0]["type"] == "hello_ok"

            finally:
                bridge.stop()
                bridge_task.cancel()
                try:
                    await bridge_task
                except (asyncio.CancelledError, Exception):
                    pass
