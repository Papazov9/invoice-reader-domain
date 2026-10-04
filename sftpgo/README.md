# Primex feed FTP front door (SFTPGo + feed-bridge)

The supplier **Primex** can only push their product/price feed over **plain FTP**.
Railway can't host FTP (needs port 21 + a passive port range + a static IP), so
this runs on the **same VPS as the invoice reader**, as two extra Compose
services. The existing `primex-feed-import` edge function does all the parsing,
validation, pricing and atomic apply — nothing here touches the database.

```
Primex ──plain FTP (21)──▶ sftpgo ──writes──▶ sftpgo_data volume (/…/primex)
                                                   │
                                     feed-bridge polls the volume
                                                   │
                        upload ──▶ Supabase Storage: feed-quarantine
                                                   │
                        trigger ──▶ primex-feed-import ──▶ apply_primex_import()
```

## Files
- `sftpgo.json` — SFTPGo server config (SFTP disabled, FTP on 2121, passive 50000–50100, admin UI on localhost).
- `users.example.json` — template for the `primex` user. **Copy to `users.json`** (gitignored), set the password + Primex IP.
- `bridge/` — the "isolated proxy": polls the upload dir, pushes each new CSV to `feed-quarantine`, triggers the import.

## What goes in `.env` (on the VPS, never committed)
See `../.env.example`. The new keys:
- `VPS_PUBLIC_IP` — the box's public IPv4 (SFTPGo advertises it for passive FTP).
- `SUPABASE_URL` — `https://nuhhxicnqfxbgedlowjk.supabase.co` (live gumialex project).
- `SUPABASE_SERVICE_ROLE_KEY` — that project's service-role key.
- `PRIMEX_IMPORT_SECRET` — must equal the `PRIMEX_IMPORT_SECRET` secret in that Supabase project.

## Deploy
```bash
cp sftpgo/users.example.json sftpgo/users.json
# edit sftpgo/users.json: set password + "<PRIMEX_IP>/32"
# edit .env: set VPS_PUBLIC_IP, SUPABASE_*, PRIMEX_IMPORT_SECRET
docker compose up -d --build
docker compose logs -f sftpgo feed-bridge
```

## Firewall — mandatory (plain FTP = cleartext password)
Lock FTP to Primex's IP at the OS level too (defence in depth; SFTPGo also
enforces `allowed_ip`). Example with ufw:
```bash
PRIMEX_IP=1.2.3.4
sudo ufw allow from $PRIMEX_IP to any port 21 proto tcp
sudo ufw allow from $PRIMEX_IP to any port 50000:50100 proto tcp
# 8080 (admin) stays closed to the world — it's bound to 127.0.0.1; use an SSH tunnel:
#   ssh -L 8080:127.0.0.1:8080 user@vps   then open http://localhost:8080
```

## DNS / Cloudflare caveat
FTP **cannot** go through Cloudflare's proxy (orange cloud). Primex must connect
to the **raw VPS IP**, or to a DNS record that is **not** proxied (grey cloud),
e.g. `ftp.unparalleled.bg` → A record → VPS IP, DNS-only. The `invoice-bot`
HTTPS record is unaffected.

## Onboarding Primex
Give them only: **host** (VPS IP or the grey-cloud DNS name), **port 21**,
**username `primex`**, **password**, mode **passive**. Ask them for their
**source IP** first (for the allowlist) and whether they can do **FTPS**.

## Optional upgrade to FTPS (encrypt the login)
If Primex supports FTPS, stop sending the password in cleartext:
1. Put a cert on the box (e.g. Let's Encrypt for `ftp.unparalleled.bg`).
2. In `sftpgo.json` set the ftpd binding `"tls_mode": 2` and
   `"certificate_file"` / `"certificate_key_file"` to the cert paths (mount them in).
3. `docker compose up -d`. Same ports; now the control channel is encrypted.
The IP allowlist stays regardless.

## Verifying
- Upload a sample CSV as `primex` → within ~30–50s it should appear in
  `feed-quarantine`, the import runs, and the local file moves to
  `…/primex/processed/`. Failures move to `…/primex/failed/` with a `.json` note.
- `docker compose logs -f feed-bridge` shows each step.
