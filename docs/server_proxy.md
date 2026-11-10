# Server proxy

Configured 2026-10-12 on the existing server as user `star`.

- Executable: `/home/star/.local/bin/mihomo`, official release v1.19.32.
- Subscription and runtime configuration: `/home/star/.config/agentic-proxy/`.
  These files contain credentials and are not copied into the repository.
- HTTP/SOCKS mixed listener: `127.0.0.1:17890`.
- Authenticated local controller: `127.0.0.1:19090`.
- User service: `agentic-proxy.service`, enabled with user lingering enabled.
- No system-wide proxy, TUN device, firewall or H3 environment changes.

Run an agent command through the proxy:

```bash
cd /home/star/agentic_world
sh scripts/with_proxy.sh .venv/bin/python scripts/audit_scene_alignment.py --help
```

The wrapper applies HTTP_PROXY and HTTPS_PROXY only to that process tree.
Existing already-running processes are not redirected. Use the wrapper for
future prototype commands that need this route.

```bash
systemctl --user status agentic-proxy
systemctl --user restart agentic-proxy
systemctl --user stop agentic-proxy
```

The subscription currently supplies 46 nodes. The first node passed the target
gateway connectivity check and is selected. This is a tested working node, not
a claim that every node was benchmarked. Subscription refresh is manual.

Real `gpt-6-astra` text requests on the same gateway succeeded both directly
(HTTP 200, 11.30 seconds) and through the proxy (HTTP 200, 9.64 seconds).
These single samples establish connectivity, not a latency benchmark or proof
that earlier gateway 502 responses were caused by the direct network route.
Results: `outputs/proxy_check_20261012/api_check.json`.
