# Beni Cloudflare Worker WebSocket Relay

This Cloudflare Worker provides a **100% free forever**, permanent, zero-trust WebSocket bridge between your Jetson Nano robot and the Kaggle Cloud Brain.

Neither machine needs port forwarding, public IP addresses, or VPN daemons (`tailscaled`), completely bypassing Kaggle's VPN detection.

---

## 2-Minute Setup: Web Dashboard (No Tools Needed, 100% Free)

You do **not** need a credit card, domain name, or local installation.

1. **Sign up or log in**: Go to [dash.cloudflare.com](https://dash.cloudflare.com) (free account).
2. **Create Worker**:
   - In the left sidebar, click **Compute (Workers) > Workers & Pages**.
   - Click **Create Application** -> **Create Worker**.
   - Name it `beni-relay` and click **Deploy**.
3. **Paste the Code**:
   - On the worker page, click **Edit code**.
   - Delete everything in `worker.js`, copy the full contents of [`worker.js`](worker.js), and paste it in.
   - Click **Save and deploy**.
4. **(Optional & Recommended) Enable Durable Objects**:
   - Go back to the Worker page -> **Settings** -> **Bindings**.
   - Under **Durable Object Bindings**, click **Add binding**.
   - Variable name: `RELAY`
   - Durable Object class: `BeniRelay`
   - Click **Save and deploy**.
   *(Note: Cloudflare includes Durable Objects in the Free plan. If not enabled, the worker runs in single-isolate mode).*
5. **(Optional) Add your secret token**:
   - In **Settings** -> **Variables and Secrets**.
   - Click **Add** under *Environment Variables*.
   - Variable name: `BENI_TOKEN`, value: your `BENI_TOKEN` from `/etc/beni/beni.env`.
   - Click **Save and deploy**.

Your relay is now live at:
```text
https://beni-relay.<your-subdomain>.workers.dev
```

---

## Setup with Wrangler CLI (Automated Migration)

Cloudflare Free Plan Durable Objects require SQLite-backed storage (`new_sqlite_classes` migration). Wrangler manages this automatically:

```bash
cd cloudflare
npx wrangler deploy
```

To set the token secret:
```bash
npx wrangler secret put BENI_TOKEN
```

---

## Testing & Verification

1. **Check Relay Status in Browser**:
   Open `https://beni-relay.<your-subdomain>.workers.dev` in any browser or run:
   ```bash
   curl https://beni-relay.<your-subdomain>.workers.dev/health
   ```
   You will see:
   ```json
   {
     "status": "ok",
     "brain_connected": false,
     "robot_connected": false,
     "frames_relayed": 0
   }
   ```

2. **Connecting the Cloud Brain (Kaggle)**:
   In your Kaggle notebook Secrets, add:
   - Secret key: `BENI_RELAY_URL`
   - Value: `wss://beni-relay.<your-subdomain>.workers.dev`
   The gateway connects outbound to `/brain`.

3. **Connecting the Robot (Jetson Nano)**:
   In `/etc/beni/beni.env` on your Jetson Nano:
   ```ini
   BENI_BRAIN_URL=wss://beni-relay.<your-subdomain>.workers.dev/robot
   ```
   Restart the agent:
   ```bash
   sudo systemctl restart beni-agent
   ```

4. Once both are running, `curl https://.../health` will report both `brain_connected: true` and `robot_connected: true`!
