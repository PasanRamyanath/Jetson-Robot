"""Cloudflare Worker WebSocket Relay Bridge (§9.1, §9.2).

Bridges the local brain gateway (:8765) to a remote WebSocket relay (e.g. Cloudflare Worker)
via pure outbound connections. This completely replaces VPN tools like tailscaled on Kaggle,
avoiding container termination while retaining a fixed permanent URL.
"""
import asyncio
import logging
import websockets
from websockets.exceptions import ConnectionClosed

log = logging.getLogger("relay")
MAX_FRAME = 8 << 20
RECONNECT_BASE_S = 1.0
RECONNECT_MAX_S = 30.0


def _is_closed(ws):
    if ws is None:
        return True
    st = getattr(ws, "state", None)
    if st is not None:
        return getattr(st, "name", "") in ("CLOSING", "CLOSED") or st in (2, 3)
    return getattr(ws, "closed", False)


class RelayBridge:
    """Manages outbound WebSocket connections from the Kaggle brain to the Cloudflare Worker relay."""

    def __init__(self, relay_url: str, token: str = "", local_port: int = 8765):
        # Normalize relay URL (wss:// or ws://)
        u = relay_url.strip()
        if u.startswith("https://"):
            u = "wss://" + u[8:]
        elif u.startswith("http://"):
            u = "ws://" + u[7:]
        elif not u.startswith(("ws://", "wss://")):
            u = "wss://" + u
        self.relay_url = u.rstrip("/")
        self.token = token
        self.local_port = local_port
        self._stopped = asyncio.Event()

    def stop(self):
        self._stopped.set()

    async def run(self):
        """Run both real-time and bulk bridges concurrently."""
        log.info("starting relay bridge to %s (local port %d)", self.relay_url, self.local_port)
        await asyncio.gather(
            self._run_realtime_loop(),
            self._run_bulk_loop(),
            return_exceptions=True
        )

    # ------------------------------------------------------------------ Real-time Bridge
    async def _run_realtime_loop(self):
        backoff = RECONNECT_BASE_S
        endpoint = f"{self.relay_url}/brain"
        if self.token:
            endpoint += f"?token={self.token}"

        while not self._stopped.is_set():
            try:
                log.info("dialing relay real-time endpoint: %s", self.relay_url + "/brain")
                async with websockets.connect(
                    endpoint,
                    max_size=MAX_FRAME,
                    ping_interval=20,
                    ping_timeout=20,
                    compression=None,
                    open_timeout=10,
                    close_timeout=5,
                ) as ws_relay:
                    log.info("connected to relay %s/brain", self.relay_url)
                    backoff = RECONNECT_BASE_S
                    await self._bridge_realtime_stream(ws_relay)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("relay real-time connection error: %s (reconnecting in %.1fs)", e, backoff)

            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=backoff)
                break
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, RECONNECT_MAX_S)

    async def _bridge_realtime_stream(self, ws_relay):
        """Pumps frames between ws_relay and a local gateway connection.
        When a robot connects, a local gateway session is established.
        When the robot disconnects or restarts, the local session cycles cleanly."""
        ws_local = None
        pump_task = None

        try:
            async for raw in ws_relay:
                # Check for relay control notices
                if isinstance(raw, str) and "peer_disconnected" in raw:
                    log.info("robot disconnected from relay")
                    if ws_local:
                        await ws_local.close()
                        ws_local = None
                    if pump_task:
                        pump_task.cancel()
                        pump_task = None
                    continue

                # If local connection is not active, connect to the local gateway
                if _is_closed(ws_local):
                    try:
                        ws_local = await websockets.connect(
                            f"ws://127.0.0.1:{self.local_port}/ws",
                            max_size=MAX_FRAME,
                            ping_interval=20,
                            ping_timeout=20,
                            compression=None,
                            open_timeout=5,
                            close_timeout=5
                        )
                        pump_task = asyncio.create_task(self._pump_local_to_relay(ws_local, ws_relay))
                        log.info("opened local session on :%d/ws for incoming relay frame", self.local_port)
                    except Exception as e:
                        log.error("failed to connect to local gateway on :%d/ws: %s", self.local_port, e)
                        continue

                # Forward frame from relay to local gateway
                try:
                    await ws_local.send(raw)
                except ConnectionClosed:
                    if ws_local:
                        await ws_local.close()
                        ws_local = None
                    if pump_task:
                        pump_task.cancel()
                        pump_task = None
        finally:
            if pump_task:
                pump_task.cancel()
            if not _is_closed(ws_local):
                await ws_local.close()

    async def _pump_local_to_relay(self, ws_local, ws_relay):
        """Forward frames from local gateway back to Cloudflare relay."""
        try:
            async for raw in ws_local:
                await ws_relay.send(raw)
        except (ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception as e:
            log.debug("pump local->relay error: %s", e)

    # ------------------------------------------------------------------ Bulk Bridge
    async def _run_bulk_loop(self):
        backoff = RECONNECT_BASE_S
        endpoint = f"{self.relay_url}/brain/bulk"
        if self.token:
            endpoint += f"?token={self.token}"

        while not self._stopped.is_set():
            try:
                async with websockets.connect(
                    endpoint,
                    max_size=MAX_FRAME,
                    ping_interval=20,
                    ping_timeout=20,
                    compression=None,
                    open_timeout=10,
                    close_timeout=5,
                ) as ws_relay:
                    log.info("connected to relay bulk channel %s/brain/bulk", self.relay_url)
                    backoff = RECONNECT_BASE_S
                    await self._bridge_bulk_stream(ws_relay)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug("relay bulk connection: %s (reconnecting in %.1fs)", e, backoff)

            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=backoff)
                break
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, RECONNECT_MAX_S)

    async def _bridge_bulk_stream(self, ws_relay):
        """Bridges bulk transfers from relay to local :8765/bulk."""
        ws_local = None
        pump_task = None
        try:
            async for raw in ws_relay:
                if isinstance(raw, str) and "peer_disconnected" in raw:
                    if ws_local:
                        await ws_local.close()
                        ws_local = None
                    if pump_task:
                        pump_task.cancel()
                        pump_task = None
                    continue

                if _is_closed(ws_local):
                    try:
                        ws_local = await websockets.connect(
                            f"ws://127.0.0.1:{self.local_port}/bulk",
                            max_size=MAX_FRAME,
                            ping_interval=20,
                            ping_timeout=20,
                            compression=None,
                            open_timeout=5,
                            close_timeout=5
                        )
                        pump_task = asyncio.create_task(self._pump_local_to_relay(ws_local, ws_relay))
                    except Exception as e:
                        log.error("failed to connect to local gateway on :%d/bulk: %s", self.local_port, e)
                        continue

                try:
                    await ws_local.send(raw)
                except ConnectionClosed:
                    if ws_local:
                        await ws_local.close()
                        ws_local = None
                    if pump_task:
                        pump_task.cancel()
                        pump_task = None
        finally:
            if pump_task:
                pump_task.cancel()
            if not _is_closed(ws_local):
                await ws_local.close()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Run standalone Cloudflare Worker relay bridge for Beni Brain")
    ap.add_argument("--relay", required=True, help="Relay URL, e.g. wss://beni-relay.<user>.workers.dev")
    ap.add_argument("--token", default="", help="Shared Beni token")
    ap.add_argument("--port", type=int, default=8765, help="Local gateway port (default 8765)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s")
    bridge = RelayBridge(args.relay, args.token, args.port)
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
