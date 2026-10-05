# CLAUDE.md

This is a home robot ("Beni"). A Jetson Nano runs the robot, and a Kaggle 2×T4 kernel runs the cloud brain. `Beni_Robot_Engineering_Blueprint.md` is the spec: the `§N` references in code point to its sections. Read the relevant section before changing behaviour.

## Python versions (hard constraints)
- `shared/beni_common/{proto,schemas,hlc,tegrastats}.py`: **Python 3.6-safe**. The Jetson host vision process imports them, so no f-strings with `=`, no walrus, no dataclasses.
- `shared/beni_common/memory/*` and `jetson/agent`: **Python 3.8**. No `match`, no `X | Y` types, no `list[int]`. No `asyncio.to_thread` (3.9+): use `loop.run_in_executor`.
- `kaggle/beni_brain`: Python 3.12.

## Commands
- `make test`: all tests; the brain runs in stub mode (`BENI_STUB=1`), with no GPU or network.
- `make lint`: pyflakes.
- `make brain-stub`: a local brain on :8765.

## Conventions
- IPC between Jetson processes uses ZMQ with msgpack. The robot↔brain link is one websocket carrying msgpack frames (`beni_common.schemas`). Every frame has `type`, `seq`, `t`, and `turn` where relevant.
- Memory: the Jetson DB is the system of record, and the brain keeps a mirror. Every synced row carries an HLC `"%013d-%04d-node"` and merges LWW. Deletes are tombstones. Only an ack advances a sync cursor (`DeltaSync`).
- Keep the Nano lean. Prefer CPU-cheap stdlib code over new dependencies, since each process costs RAM from a shared 3.4 GB. No LangGraph or other frameworks on the brain either: the pipeline is plain asyncio.
- `asyncio.Event` and futures are not thread-safe. Work done inside `asyncio.to_thread` must not touch them. Do that on the loop afterwards (see `Mirror.activate`).
- Model stubs (`StubLLM`, `StubSTT`, `StubTTS`, `HashEmbedder`) must stay behaviour-compatible, because the e2e test depends on them.
- Match the surrounding style: short docstrings with a `§` reference, no over-commenting, 120-column lines.
