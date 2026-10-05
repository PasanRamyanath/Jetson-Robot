"""OpenAI-compatible async client for the local vLLM server (§10.5), plus a deterministic stub.

stream() yields ("text", str) deltas and, at the end, ("tools", [{id, name, arguments}]) when the model called tools.
Cancelling the consuming task closes the HTTP stream, which aborts the request inside vLLM.
"""
import asyncio
import json
import logging
import re

log = logging.getLogger("llm")


class LLM:
    def __init__(self, cfg):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=cfg.llm_url, api_key="EMPTY", timeout=60, max_retries=0)
        self.model = cfg.llm_model

    async def ready(self, timeout=900):
        """Wait for vLLM to finish loading (the gateway starts in parallel with it)."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            try:
                await self.client.models.list()
                return True
            except Exception:
                await asyncio.sleep(3)
        return False

    async def stream(self, messages, tools=None, max_tokens=300, temperature=0.7):
        kw = {"tools": tools, "tool_choice": "auto"} if tools else {}
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, max_tokens=max_tokens, temperature=temperature, **kw)
        calls = {}
        try:
            async for ch in resp:
                if not ch.choices:
                    continue
                d = ch.choices[0].delta
                if d.content:
                    yield "text", d.content
                for tc in d.tool_calls or ():
                    c = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                    c["id"] = tc.id or c["id"]
                    if tc.function:
                        c["name"] += tc.function.name or ""
                        c["arguments"] += tc.function.arguments or ""
        finally:
            await resp.close()
        if calls:
            yield "tools", [dict(c, id=c["id"] or "call_%d" % i) for i, c in sorted(calls.items())]

    async def json(self, prompt, schema, max_tokens=512, system=None):
        """Structured output (vLLM guided decoding) -> dict ({} on failure)."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        try:
            r = await self.client.chat.completions.create(
                model=self.model, messages=msgs, max_tokens=max_tokens, temperature=0.0,
                response_format={"type": "json_schema", "json_schema": {"name": "out", "schema": schema}})
            return json.loads(r.choices[0].message.content or "{}")
        except Exception as e:
            log.warning("json call failed: %s", e)
            return {}

    async def complete(self, prompt, max_tokens=300, temperature=0.3):
        r = await self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
            temperature=temperature)
        return (r.choices[0].message.content or "").strip()

    async def warmup(self, system):
        """Builds the prefix cache for the system prompt (§10.3)."""
        try:
            await self.client.chat.completions.create(
                model=self.model, max_tokens=1,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": "hi"}])
        except Exception as e:
            log.warning("warm-up failed: %s", e)


class StubLLM:
    """Echo model for tests: '<emo=happy>You said: X.'; 'spin' in the text triggers a look_at tool call first."""

    async def ready(self, timeout=0):
        return True

    async def warmup(self, system):
        pass

    async def stream(self, messages, tools=None, max_tokens=300, temperature=0.7):
        last = messages[-1]
        if tools and last["role"] == "user" and re.search(r"\bspin\b", _text(last)):
            yield "tools", [{"id": "call_0", "name": "look_at", "arguments": '{"target": "left"}'}]
            return
        said = next((_text(m) for m in reversed(messages) if m["role"] == "user"), "")
        for w in ("<emo=happy>", "You ", "said: ", said.strip()[:200], "."):
            await asyncio.sleep(0)
            yield "text", w

    async def json(self, prompt, schema, max_tokens=512, system=None):
        return {"facts": [], "importance": 3, "episode_summary": prompt[-120:], "op": "NOOP"}

    async def complete(self, prompt, max_tokens=300, temperature=0.3):
        return prompt[-200:]


def _text(m):
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if p.get("type") == "text")
    return c or ""


def load(cfg):
    return StubLLM() if cfg.stub else LLM(cfg)
