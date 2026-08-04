# Autonomous Retail Checkout System

## Safe setup for local or public use

1. Keep your private camera settings out of git.
2. Put real camera values in `config.local.json`.
3. Copy `config.example.json` to `config.local.json` and edit that file locally.
4. Set these environment variables before running in shared or public environments:
   - `FLASK_SECRET_KEY`
   - `ADMIN_USER`
   - `ADMIN_PASS`
5. Do not commit `config.local.json`.

## Camera configuration

- `config.json` contains public-safe placeholder values.
- The admin panel saves live settings to `config.local.json`.
- The app normalizes base IP Webcam URLs like `http://HOST:8080` to `http://HOST:8080/video`.
