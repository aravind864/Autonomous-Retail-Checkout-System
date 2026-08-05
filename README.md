# Autonomous Retail Checkout System

## Configuration & Usage

1. Copy `.env.example` to `.env` to configure your environment secrets:
   - `FLASK_SECRET_KEY`
   - `ADMIN_USER` (Default: `admin`)
   - `ADMIN_PASS` (Default: `admin`)
   - `HOST_IP`
   - `HOST_PORT`

2. Camera Configuration:
   - Camera sources are saved directly to `config.json` via the Admin Panel at `http://localhost:5000/admin`.
   - Camera settings update live instantly in the Surveillance Dashboard without requiring a server restart.
   - The application normalizes base IP Webcam URLs (e.g. `http://HOST:8080` to `http://HOST:8080/video`).

