"""§9.2 cloud link: a malformed frame from the brain is dropped, the link stays up and keeps delivering frames."""
import asyncio

import msgpack
import websockets

from beni_agent.cloud.link import CloudLink


def test_malformed_frame_does_not_kill_the_link():
    got = []

    async def brain(ws, *_):
        await ws.recv()                                                   # hello
        await ws.send(msgpack.packb({"type": "hello_ok"}))
        await ws.send(b"\xc1 not msgpack")                               # 0xc1 is never valid msgpack
        await ws.send(msgpack.packb([1, 2]))                             # valid msgpack, not a frame
        await ws.send(msgpack.packb({"type": "scene", "caption": "ok"}))
        await asyncio.sleep(1)

    async def on_frame(f):
        got.append(f)

    async def main():
        async with websockets.serve(brain, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            link = CloudLink(["ws://127.0.0.1:%d/ws" % port], "t", on_frame)
            task = asyncio.ensure_future(link.run())
            for _ in range(50):
                if got:
                    break
                await asyncio.sleep(0.05)
            assert link.online.is_set() and not task.done()
            task.cancel()
    asyncio.run(main())
    assert got == [{"type": "scene", "caption": "ok"}]
