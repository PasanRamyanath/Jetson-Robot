/**
 * Cloudflare Worker: Zero-Trust WebSocket Relay for Beni Robot (§9.1, §9.2)
 *
 * Provides a 100% free, permanent, outbound-only WebSocket relay between the Jetson Nano
 * and the Kaggle Cloud Brain. Neither end requires port forwarding, public IPs, or VPN daemons.
 *
 * Endpoints:
 *   GET /health         -> JSON status (brain_connected, robot_connected, frames_relayed)
 *   GET /               -> Status page and connection info
 *   WSS /robot, /ws     -> Robot real-time session
 *   WSS /brain          -> Brain real-time session
 *   WSS /robot/bulk     -> Robot bulk snapshot/exemplar upload channel
 *   WSS /brain/bulk     -> Brain bulk channel
 *
 * Compatible with Cloudflare Workers Free Tier (both Durable Objects and in-memory fallback).
 */

import { DurableObject } from "cloudflare:workers";

class RelayCore {
  constructor(env) {
    this.env = env || {};
    this.sessions = new Map(); // 'brain', 'robot', 'brain_bulk', 'robot_bulk'
    this.stats = {
      relayed: 0,
      startedAt: Date.now(),
      connections: 0
    };
  }

  async fetch(request) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    const upgradeHeader = request.headers.get("Upgrade");

    // 1. Health & status endpoints (HTTP)
    if (!upgradeHeader || upgradeHeader.toLowerCase() !== "websocket") {
      if (path === "/health") {
        return new Response(JSON.stringify({
          status: "ok",
          brain_connected: this.hasActiveSession("brain"),
          robot_connected: this.hasActiveSession("robot"),
          brain_bulk_connected: this.hasActiveSession("brain_bulk"),
          robot_bulk_connected: this.hasActiveSession("robot_bulk"),
          frames_relayed: this.stats.relayed,
          total_connections: this.stats.connections,
          uptime_s: Math.floor((Date.now() - this.stats.startedAt) / 1000)
        }, null, 2), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache, no-store, must-revalidate"
          }
        });
      }

      // Root landing page
      const brainOk = this.hasActiveSession("brain");
      const robotOk = this.hasActiveSession("robot");
      const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Beni Robot Cloud Relay</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; max-width: 680px; margin: 0 auto; line-height: 1.5; }
    h1 { color: #38bdf8; display: flex; align-items: center; gap: 0.5rem; }
    .card { background: #1e293b; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; border: 1px solid #334155; }
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-on { background: #065f46; color: #34d399; }
    .badge-off { background: #450a0a; color: #f87171; }
    .stat { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #334155; }
    code { background: #0f172a; padding: 0.2rem 0.4rem; border-radius: 4px; color: #38bdf8; font-size: 0.9em; }
  </style>
</head>
<body>
  <h1>🤖 Beni Robot WebSocket Relay</h1>
  <p>Zero-Trust Outbound WebSocket Bridge between Jetson Nano and Kaggle Cloud Brain.</p>
  <div class="card">
    <div class="stat"><span>Robot (Jetson Nano)</span> <span class="badge ${robotOk ? 'badge-on' : 'badge-off'}">${robotOk ? 'CONNECTED' : 'DISCONNECTED'}</span></div>
    <div class="stat"><span>Brain (Kaggle T4)</span> <span class="badge ${brainOk ? 'badge-on' : 'badge-off'}">${brainOk ? 'CONNECTED' : 'DISCONNECTED'}</span></div>
    <div class="stat"><span>Frames Relayed</span> <strong>${this.stats.relayed}</strong></div>
    <div class="stat"><span>Uptime</span> <span>${Math.floor((Date.now() - this.stats.startedAt) / 1000)}s</span></div>
  </div>
  <div class="card">
    <h3>Endpoints</h3>
    <p>Robot: <code>${url.origin}/robot</code> (or <code>${url.origin}/ws</code>)</p>
    <p>Brain: <code>${url.origin}/brain</code></p>
    <p>Health: <code>${url.origin}/health</code></p>
  </div>
</body>
</html>`;
      return new Response(html, {
        status: 200,
        headers: { "Content-Type": "text/html; charset=utf-8" }
      });
    }

    // 2. Determine WebSocket Role
    let role = "robot";
    if (path === "/brain/bulk" || (path === "/bulk" && url.searchParams.get("role") === "brain")) {
      role = "brain_bulk";
    } else if (path === "/robot/bulk" || path === "/bulk") {
      role = "robot_bulk";
    } else if (path === "/brain" || url.searchParams.get("role") === "brain") {
      role = "brain";
    } else if (path === "/robot" || path === "/ws" || url.searchParams.get("role") === "robot") {
      role = "robot";
    }

    // 3. Optional Token Authentication
    const configuredToken = this.env.BENI_TOKEN;
    if (configuredToken) {
      const qToken = url.searchParams.get("token");
      const hToken = request.headers.get("X-Beni-Token") || 
                     (request.headers.get("Authorization") || "").replace(/^Bearer\s+/i, "");
      const clientToken = qToken || hToken;
      if (!clientToken || clientToken !== configuredToken) {
        return new Response("Unauthorized: invalid or missing token", { status: 401 });
      }
    }

    // 4. Accept WebSocket connection
    const webSocketPair = new WebSocketPair();
    const [client, server] = Object.values(webSocketPair);
    server.accept();

    this.handleSession(server, role);

    return new Response(null, {
      status: 101,
      webSocket: client
    });
  }

  hasActiveSession(role) {
    const ws = this.sessions.get(role);
    return ws !== undefined && ws !== null && ws.readyState === WebSocket.OPEN;
  }

  handleSession(ws, role) {
    this.stats.connections++;

    // Close any previous session for this exact role
    const existing = this.sessions.get(role);
    if (existing && existing.readyState === WebSocket.OPEN) {
      try {
        existing.close(1000, "Replaced by new connection");
      } catch (_) {}
    }
    this.sessions.set(role, ws);

    // Partner mapping
    const partnerRole = 
      role === "brain" ? "robot" :
      role === "robot" ? "brain" :
      role === "brain_bulk" ? "robot_bulk" :
      "brain_bulk";

    // Message forwarder
    ws.addEventListener("message", event => {
      const partner = this.sessions.get(partnerRole);
      if (partner && partner.readyState === WebSocket.OPEN) {
        partner.send(event.data);
        this.stats.relayed++;
      }
    });

    const cleanup = () => {
      if (this.sessions.get(role) === ws) {
        this.sessions.delete(role);
        // Notify partner that peer disconnected
        const partner = this.sessions.get(partnerRole);
        if (partner && partner.readyState === WebSocket.OPEN) {
          try {
            partner.send(JSON.stringify({ type: "peer_disconnected", role }));
          } catch (_) {}
        }
      }
    };

    ws.addEventListener("close", cleanup);
    ws.addEventListener("error", cleanup);
  }
}

/**
 * Cloudflare Durable Object class.
 * Exported so Cloudflare Dashboard detects it and populates the Durable Object dropdown.
 */
export class BeniRelay extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.core = new RelayCore(env);
  }

  async fetch(request) {
    return this.core.fetch(request);
  }
}

// Fallback in-memory instance for standalone Workers before Durable Object binding is set
let fallbackCore = null;

export default {
  async fetch(request, env, ctx) {
    // If Durable Object binding is configured, route to global singleton room
    if (env && env.RELAY) {
      const id = env.RELAY.idFromName("beni-global-relay");
      const obj = env.RELAY.get(id);
      return obj.fetch(request);
    }

    // Fallback: in-memory instance within current Worker isolate
    if (!fallbackCore) {
      fallbackCore = new RelayCore(env);
    }
    return fallbackCore.fetch(request);
  }
};
