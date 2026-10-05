"""System prompt, tool schemas and memory-pipeline prompts (§10.6, §11.7, §11.9)."""
import json

EMOS = ("happy", "excited", "sad", "calm", "curious", "apologetic", "neutral")

SYSTEM = """You are Beni, a small home companion robot in {home_city}, Sri Lanka. You have a body: wheels, a head with two \
cameras, eye displays, a speaker. Personality: {persona}.
Rules: Reply in 1-3 short spoken sentences unless asked for more. Start every reply with <emo=X> where X is one of \
{emos}. Never read out lists, markdown or emoji. Use tools to act; never claim you did something physical unless the \
tool result says ok. If unsure what the user refers to, look (take_snapshot) before answering. Use names sparingly.
Current context: time {time}; place {place}; battery {battery}%; people present: {people}; unknown faces: {unknown}; \
robot: {robot}; phones at home: {home}.
Recent events: {events}
Scene: {scene}
{memory}"""

PROACTIVE = {
    "greet": "You just noticed {person} arrive. Greet them warmly in one short sentence; mention something you "
             "remember about them only if it is natural.",
    "remind": "It is time to remind {person}: \"{text}\". Say it kindly in one sentence.",
    "ask_name": "Someone you don't recognise is here. Politely introduce yourself and ask their name in one sentence.",
    "dock": "Your battery is low ({battery}%). Say briefly that you are going to charge.",
    "routine_nudge": "{person} usually chats with you around this time. Check in with them in one short, warm "
                     "sentence.",
    "share_memory": "Share this memory with {person} in one or two warm sentences, as 'a year ago today': {text}",
    "lost_item": "Tell {person} in one sentence: {text}",
    "ask_visitor": "Ask {person} politely, in one sentence: {text}",
}


def _fn(name, desc, props=None, req=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": list(req)}}}


_S = {"type": "string"}
TOOLS = [
    _fn("move_to", "Navigate to a known place", {"place": _S}, ["place"]),
    _fn("look_at", "Turn head/body toward a person, object or direction", {"target": _S}, ["target"]),
    _fn("follow_person", "Follow a person", {"name": _S}, ["name"]),
    _fn("come_here", "Come to the person who is speaking"),
    _fn("stop", "Stop all motion now"),
    _fn("set_expression", "Show an eye expression", {"expression": {"type": "string", "enum": [
        "happy", "sad", "surprised", "sleepy", "curious", "love", "angry", "wink"]}}, ["expression"]),
    _fn("take_snapshot", "Look through a camera; returns an image you can see",
        {"camera": {"type": "string", "enum": ["head", "front"]}}),
    _fn("locate", "Find something in the camera view and turn to look at it ('where is the red cup?', "
        "'look at the door')", {"object": _S, "camera": {"type": "string", "enum": ["head", "front"]}},
        ["object"]),
    _fn("find_object", "Search memory (and optionally the room) for where an object is",
        {"object": _S, "search_room": {"type": "boolean"}}, ["object"]),
    _fn("remember", "Store an explicit fact the user asked you to remember", {"fact": _S, "about": _S}, ["fact"]),
    _fn("recall", "Search long-term memory. time_range: 'today', 'yesterday', 'this week', 'last 3 days'...",
        {"query": _S, "about": _S, "time_range": _S}, ["query"]),
    _fn("set_reminder", "Remind someone at a time. when: ISO time '2026-09-27T18:30', 'HH:MM' or 'in N minutes/hours'",
        {"who": _S, "when": _S, "what": _S}, ["who", "when", "what"]),
    _fn("enroll_person", "Learn the face of the person in front of you", {"name": _S}, ["name"]),
    _fn("forget_me", "Forget everything about the person speaking (only when they explicitly ask)"),
    _fn("goto_dock", "Go charge"),
    _fn("save_place", "Remember where you are standing now under a name (e.g. 'kitchen'; 'dock' for the charger)",
        {"name": _S}, ["name"]),
    _fn("camera_privacy", "Privacy mode: cameras off and eyes closed (on=true), or back on (on=false); only when "
        "asked", {"on": {"type": "boolean"}}, ["on"]),
]
PHYSICAL = {"move_to", "look_at", "follow_person", "come_here", "stop", "set_expression", "set_reminder",
            "enroll_person", "goto_dock", "save_place", "camera_privacy"}


def _names(people):
    return ", ".join(p.get("name") or "unknown" for p in people) or "nobody"


def system(cfg, ctx, memory_text):
    return SYSTEM.format(
        home_city=cfg.home_city, persona=cfg.persona, emos=", ".join(EMOS),
        time=ctx.get("time", "?"), place=ctx.get("place") or "unknown", battery=ctx.get("battery", "?"),
        people=_names(ctx.get("people_present", [])), unknown=ctx.get("unknown_faces", 0),
        robot=json.dumps(ctx.get("robot_state") or {}, separators=(",", ":")),
        home=", ".join(ctx.get("phones_home") or ()) or "unknown",
        events="; ".join(ctx.get("recent_events") or []) or "none",
        scene=ctx.get("last_scene_caption") or "unknown",
        memory=memory_text or "")


def proactive_instruction(p, ctx):
    tpl = PROACTIVE.get(p.get("behaviour"), "Say something brief and friendly.")
    return "[No one spoke. Proactive behaviour] " + tpl.format(
        person=p.get("name") or p.get("person") or "someone", text=p.get("text", ""),
        battery=ctx.get("battery", "?"))


# ------------------------------------------------------------------ memory pipeline (§11.7)
EXTRACT = """From this conversation turn, extract durable facts worth remembering about people, the home, or Beni itself. \
Skip small talk and temporary states (unless time-bound, then give valid_to). Explicit instructions about how Beni \
must behave ("don't come into my room", "always speak Sinhala with Achchi", "no scary stories for the kids") are facts \
with predicate "rule", subject = the person concerned (or home), text = the rule in third person.
Speaker: {speaker_name} (id {speaker_id}). Present: {people}. Time: {now}.
Turn:
USER: {user}
BENI: {beni}"""

EXTRACT_SCHEMA = {"type": "object", "properties": {
    "facts": {"type": "array", "items": {"type": "object", "properties": {
        "subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"},
        "text": {"type": "string"}, "confidence": {"type": "number"},
        "valid_to": {"type": ["string", "null"]},
        "source_kind": {"type": "string", "enum": ["said_by_user", "told_by_other", "inferred"]}},
        "required": ["subject", "predicate", "object", "text", "confidence", "source_kind"]}},
    "importance": {"type": "integer"},
    "episode_summary": {"type": "string"},
    "feedback": {"type": ["object", "null"], "properties": {
        "kind": {"type": "string", "enum": ["correction", "praise", "complaint"]}, "about": {"type": "string"}}}},
    "required": ["facts", "importance", "episode_summary"]}

DECIDE = """Existing memories (id: text):
{existing}
New candidate: {candidate}
Decide one: ADD (new info), UPDATE <id> (same topic, newer/more precise info), DELETE <id> (candidate contradicts and \
retracts it), NOOP (already known or not useful)."""

DECIDE_SCHEMA = {"type": "object", "properties": {
    "op": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NOOP"]},
    "target": {"type": ["string", "null"]}, "merged_text": {"type": ["string", "null"]}},
    "required": ["op"]}

CARD = """Write a core-memory card (max 600 characters, plain prose, third person) about {name} for a home robot, from \
these facts and reflections. Keep stable facts (relation, preferences, routines, important dates); drop trivia.
{facts}"""

SUMMARY = """Summarise this {span} ({when}) at home for a home robot's long-term memory, in at most 4 short \
sentences of plain third-person prose. Keep who did what, plans, feelings and anything worth remembering; drop routine \
sightings unless they show a pattern.
{items}"""

REFLECT = """Recent memories about {name}, numbered:
{items}
What 3 high-level insights can be inferred about {name}? For each, cite the numbers of the memories it rests on \
and, if it is a durable fact (a preference, habit, relationship or interest), give it as predicate/object too."""

REFLECT_SCHEMA = {"type": "object", "properties": {
    "insights": {"type": "array", "items": {"type": "object", "properties": {
        "text": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "integer"}},
        "predicate": {"type": ["string", "null"]}, "object": {"type": ["string", "null"]}},
        "required": ["text", "evidence"]}}},
    "required": ["insights"]}

SENSITIVE = ("health", "medical", "illness", "religion", "relationship", "finance", "money", "salary", "politic")
