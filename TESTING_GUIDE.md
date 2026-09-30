# ShopFlow: End-to-End System Setup & Testing Guide
**Autonomous AI Retail & Frictionless In-Store Checkout System**

---

## 📌 1. System Overview & Architecture

ShopFlow is an autonomous retail checkout system combining a native Android mobile application, a Flask-powered computer vision backend, a PostgreSQL database with vector search (`pgvector`), and dual-camera real-time AI tracking (YOLO + ByteTrack + InsightFace).

### 🔗 Repositories & Local Paths

| Component | GitHub Repository | Local Working Directory | Primary Tech Stack |
|---|---|---|---|
| **Android Mobile App** | [vprayag2005/shopflow-mobile](https://github.com/vprayag2005/shopflow-mobile) | `C:\Users\praya\AndroidStudioProjects\ShopFlow` | Android (Java), CameraX, ML Kit Face Detection, OkHttp |
| **Flask AI Backend** | [aravind864/Autonomous-Retail-Checkout-System](https://github.com/aravind864/Autonomous-Retail-Checkout-System) | `C:\Project\Autonomous-Retail-Checkout-System` | Python 3.10+, Flask, SQLAlchemy, OpenCV, YOLO, InsightFace, pgvector |

### 🔄 How the Entire System Interacts

```
+-------------------------------------------------------------+
|                     Android Mobile App                      |
|  1. Register Customer -> POST /api/customers/register       |
|  2. Customer ID Generated (e.g. CUST000001)                 |
|  3. Login & Face Enrollment (Front, Left, Right Poses)       |
|  4. Upload Face Embeddings -> POST /api/face/enroll         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     Flask AI Backend                        |
|  - Stores customer credentials & 512-d embeddings in DB     |
|  - Camera 1 (Entry Cam): YOLO Person Detection + Face ID    |
|  - Queries pgvector for cosine similarity >= threshold      |
|  - Identifies customer -> Labels Bounding Box: CUST000001   |
|  - Camera 2 (Shelf Cam): Tracks customer interaction        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             Web Surveillance & Admin Dashboard              |
|  - URL: http://localhost:5000/                              |
|  - Live video stream shows bounding box with Customer ID    |
|  - VERIFICATION: ID on camera matches Mobile App ID!        |
+-------------------------------------------------------------+
```

---

## 💻 2. Prerequisites & Software Installation

Before cloning or running the repositories, ensure your team members have installed the following required tools on Windows:

### A. Code Editors & IDEs
1. **Visual Studio Code**:
   - Download & Install from: [https://code.visualstudio.com/](https://code.visualstudio.com/)
   - Required Extensions:
     - **Python** (`ms-python.python`)
     - **Pylance** (`ms-python.vscode-pylance`)
2. **Android Studio**:
   - Download & Install latest Android Studio (Koala / Ladybug or newer) from: [https://developer.android.com/studio](https://developer.android.com/studio)
   - During setup, install:
     - **Android SDK Platform 34 or 36**
     - **Android SDK Build-Tools**
     - **Android SDK Platform-Tools** (includes `adb.exe`)
     - **JDK 17** (bundled automatically with Android Studio)

### B. Programming Languages & Runtimes
1. **Python (Version 3.10, 3.11, or 3.12 recommended)**:
   - Download from: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - ⚠️ **CRITICAL**: Check the checkbox **"Add Python to PATH"** during installation.
2. **Git for Windows**:
   - Download from: [https://git-scm.com/download/win](https://git-scm.com/download/win)

### C. Database (PostgreSQL with `pgvector`)
ShopFlow requires PostgreSQL with the `pgvector` extension for storing and querying 512-dimensional facial embedding vectors.

**Choose ONE of the two options below:**

#### Option 1: Docker (Recommended - Quickest & Zero Configuration)
If Docker Desktop is installed, run this single command in PowerShell:
```powershell
docker run -d --name shopflow-db -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=shopflow pgvector/pgvector:pg16
```
*(This automatically provisions PostgreSQL 16 with pgvector pre-installed and creates the `shopflow` database).*

#### Option 2: Native PostgreSQL on Windows
1. Download & install PostgreSQL (version 15 or 16) from [https://www.postgresql.org/download/windows/](https://www.postgresql.org/download/windows/).
2. Remember the `postgres` superuser password you set during installation.
3. Install `pgvector`:
   - Download the precompiled Windows release matching your PostgreSQL version from [pgvector releases](https://github.com/pgvector/pgvector/releases).
   - Copy the `.dll` to `C:\Program Files\PostgreSQL\<version>\lib`.
   - Copy the `.control` and `.sql` files to `C:\Program Files\PostgreSQL\<version>\share\extension`.
4. Open **pgAdmin** or `psql` and execute:
   ```sql
   CREATE DATABASE shopflow;
   \c shopflow
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

---

## 📂 3. Cloning Repositories & Local Directory Setup

Open Windows PowerShell or Command Prompt and clone both repositories:

```powershell
# 1. Create a root projects directory (or use your preferred location)
mkdir C:\Project -ErrorAction SilentlyContinue
cd C:\Project

# 2. Clone the Flask AI Backend repository
git clone https://github.com/aravind864/Autonomous-Retail-Checkout-System.git
# Local path: C:\Project\Autonomous-Retail-Checkout-System

# 3. Clone the Android Mobile App repository
cd C:\Users\$env:USERNAME\AndroidStudioProjects
git clone https://github.com/vprayag2005/shopflow-mobile.git ShopFlow
# Local path: C:\Users\<YourUsername>\AndroidStudioProjects\ShopFlow
```

---

## 🐍 4. Flask AI Backend Setup & Execution (VS Code)

### Step 4.1: Open Project in VS Code
1. Launch **Visual Studio Code**.
2. Go to **File > Open Folder...** and select:
   `C:\Project\Autonomous-Retail-Checkout-System`
3. Open a new terminal in VS Code (**Terminal > New Terminal** or press ``Ctrl + ` ``).

### Step 4.2: Create Virtual Environment & Install Dependencies
Run the following commands in the VS Code terminal:

```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### Step 4.3: Configure Environment Variables (`.env`)
1. In the root of `Autonomous-Retail-Checkout-System`, copy `.env.example` to `.env`:
   ```powershell
   copy .env.example .env
   ```
2. Open `.env` in VS Code and fill in the values:
   ```env
   # Secret key for session security
   FLASK_SECRET_KEY=shopflow_super_secret_key_2026_production

   # Admin credentials for surveillance dashboard login
   ADMIN_USER=admin
   ADMIN_PASS=admin123

   # Enable Flask debug mode for testing
   FLASK_DEBUG=true

   # PostgreSQL Database Configuration
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DATABASE=shopflow
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   ```

### Step 4.4: Start the Flask Server
Run:
```powershell
python app.py
```
**Expected Terminal Output:**
```
[VISION] Loading CAM1 detector/tracker: yolo26n_openvino_model ...
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5000 (Press CTRL+C to quit)
```
*(On startup, Flask automatically initializes the database tables `customers` and `face_embeddings`).*

### Step 4.5: Configure Cameras in Admin Panel
1. Open your web browser and navigate to:
   **`http://localhost:5000/admin/login`**
2. Log in using the admin credentials from your `.env`:
   - **Username**: `admin`
   - **Password**: `admin123`
3. You will be redirected to the **Admin Panel** (`http://localhost:5000/admin`).
4. Set up Camera Sources:
   - **Entry / Exit Camera (Camera 1)**:
     - If using your laptop webcam: enter `0`
     - If using a second USB camera: enter `1`
     - If using an Android phone as an IP camera (e.g. via "IP Webcam" app): enter `http://<PHONE_IP>:8080/video`
   - **Shelf View Camera (Camera 2)**:
     - Enter another webcam index (e.g., `1`) or an RTSP/HTTP stream URL, or `0` for testing.
5. Click **"Save Configuration"**. The settings persist in `.camera_settings`.
6. Click **"Dashboard"** in the top navigation bar to view the live dual-camera feed at:
   **`http://localhost:5000/`**

---

## 📱 5. Mobile App Setup & Execution (Android Studio)

### Step 5.1: Open Project in Android Studio
1. Launch **Android Studio**.
2. Click **Open** (or **File > Open**) and select:
   `C:\Users\<YourUsername>\AndroidStudioProjects\ShopFlow`
3. Wait for Gradle to finish downloading dependencies and syncing project files (watch the bottom progress bar).

### Step 5.2: Configure `local.properties`
The mobile app reads its backend API endpoint directly from `local.properties`.

1. In the project root (`C:\Users\<YourUsername>\AndroidStudioProjects\ShopFlow`), create or edit `local.properties`:
   ```properties
   ## Android SDK location on your machine
   sdk.dir=C\:\\Users\\<YourUsername>\\AppData\\Local\\Android\\Sdk

   ## Backend API server URL
   BASE_URL=http://127.0.0.1:5000
   ```

> ⚠️ **IMPORTANT NETWORKING RULE FOR ANDROID:**
> The mobile app's `network_security_config.xml` explicitly permits cleartext HTTP for `localhost` and `127.0.0.1`.
> When testing with a physical device over USB, setting `BASE_URL=http://127.0.0.1:5000` combined with ADB reverse port forwarding (Step 5.3) is **100% reliable** and avoids all firewall / IP routing issues!

### Step 5.3: Prepare Physical Android Phone & Enable USB Debugging
1. On your Android phone, enable **Developer Options**:
   - Go to **Settings > About Phone**.
   - Find **Build Number** and tap it **7 times** continuously until a toast says *"You are now a developer!"*.
2. Enable **USB Debugging**:
   - Go to **Settings > System > Developer Options** (or search "Developer Options").
   - Toggle **USB Debugging** to **ON**.
3. Connect the phone to your computer using a USB cable.
4. When a popup appears on your phone asking *"Allow USB debugging?"*:
   - Check the box **"Always allow from this computer"**.
   - Tap **Allow**.
5. Set up **ADB Reverse Port Forwarding**:
   Open a terminal window on your PC and run:
   ```powershell
   adb reverse tcp:5000 tcp:5000
   ```
   *(This routes requests made to `http://127.0.0.1:5000` on the phone directly to port 5000 on your PC!)*

### Step 5.4: Install & Launch the App
1. In Android Studio, check the device dropdown at the top toolbar (next to the Play button). Your connected physical phone model should be selected.
2. Click the green **Run** button ▶ (or press `Shift + F10`).
3. Android Studio will build the APK, install it on your physical phone, and automatically open the ShopFlow app!

---

## 🧪 6. Step-by-Step End-to-End Testing Flow

Follow these exact steps with your teammates to verify complete integration between the mobile app and computer vision surveillance backend:

```
+-----------------------------------------------------------------------------------+
|                              TESTING WORKFLOW SUMMARY                             |
|                                                                                   |
|  [Mobile App]          [Mobile App]             [Mobile App]     [Flask Dashboard]|
|  Register Account ---> Login Account -------> Face Registration -> Camera 1 Match |
|  (CUST000001 created)  (Enters Enrollment)    (3 Poses Enrolled)  (ID Verified!)  |
+-----------------------------------------------------------------------------------+
```

### Action 1: Register a New Customer Account on Mobile
1. On the mobile phone screen, you will see the **Login Screen**.
2. Tap the blue **"Register"** link at the bottom (`Don't have an account? Register`).
3. Fill in the registration form:
   - **First Name**: `Jane`
   - **Last Name**: `Doe`
   - **Email**: `jane.doe@example.com`
   - **Password**: `password123`
   - **Confirm Password**: `password123`
4. Tap **"Create Account"**.
5. **Observed Result**:
   - The app sends a `POST` request to `/api/customers/register`.
   - A success toast appears: **`Account created: CUST000001`**.
   - The app navigates back to the Login screen.

### Action 2: Login to the Account
1. Enter the registered email (`jane.doe@example.com`) and password (`password123`).
2. Tap **"Sign In"**.
3. The app authenticates with `/api/auth/login` and queries `/api/face/status/CUST000001`.
4. Because the customer has not yet registered facial biometrics, the app automatically transitions to the **Face Enrollment Screen** (`FaceEnrollmentActivity`).

### Action 3: Face Biometric Enrollment (Camera Capture)
1. Grant the camera permission if prompted by the phone.
2. The biometric guide overlay will appear on the screen:
   - **Step 1 - Front Pose**: Align your face inside the oval looking straight ahead. Tap **Capture**.
   - **Step 2 - Left Profile**: Turn your head slightly to the left. Tap **Capture**.
   - **Step 3 - Right Profile**: Turn your head slightly to the right. Tap **Capture**.
3. Tap **"Submit Biometrics"**.
4. **Observed Result**:
   - The app extracts 512-dimensional embeddings and uploads them to `/api/face/enroll`.
   - The backend stores them in PostgreSQL `face_embeddings` table and marks `face_enrolled = TRUE`.
   - The app transitions to the **Customer Dashboard** (`MainActivity`).

### Action 4: Verify Customer ID Badge on Mobile
1. On the Customer Dashboard home screen, check the top profile section:
   - **Greeting**: `Hi, Jane 👋`
   - **Avatar Initials**: `JD`
   - **Customer ID Badge**: `Customer ID: CUST000001`
2. Open the navigation drawer (tap the ☰ hamburger icon at top left):
   - Check the drawer header displays: `Jane Doe`, `jane.doe@example.com`, and `Customer ID: CUST000001`.
3. Tap **Account** in the drawer:
   - Notice **Face Biometrics** status shows: `✓ Registered` (Green badge).

### Action 5: Live In-Store Identification via Flask Surveillance Dashboard
1. On your PC, ensure the Flask backend is running and open the **Surveillance Dashboard**:
   **`http://localhost:5000/`**
2. Look at the **Camera 1: Entry / Exit Cam** video panel.
3. Have the registered person (Jane) stand or walk in front of Camera 1.
4. **Observed Result in Real Time**:
   - YOLO detects the person and draws a tracking bounding box.
   - InsightFace crops the face and extracts the live 512-d embedding.
   - pgvector executes a cosine similarity search against registered customer embeddings.
   - In the terminal, Flask logs:
     ```
     [ENTRY FACE ACCEPT] Track 1 identified as CUST000001 (Jane Doe) with confidence 0.824
     ```
   - On the Surveillance Dashboard live stream, the bounding box turns green and displays:
     **`CUST000001: Jane Doe`**

### ✅ VERIFICATION CRITERIA
| Checkpoint | Target | Status |
|---|---|---|
| **Mobile Registration** | Toast shows generated ID: `CUST000001` | Pass [ ] |
| **Mobile Home Screen** | Profile badge displays `Customer ID: CUST000001` | Pass [ ] |
| **Mobile Account Screen** | Face Status shows `✓ Registered` | Pass [ ] |
| **Surveillance Dashboard** | Camera 1 detects face and labels box `CUST000001` | Pass [ ] |
| **ID Match** | Dashboard Customer ID matches Mobile App Customer ID | **SUCCESS!** [ ] |

---

## 🛠️ 7. Troubleshooting & Common Issues

| Issue / Symptom | Root Cause | Solution |
|---|---|---|
| **App says "Unable to connect to server"** | Mobile phone cannot route to PC's port 5000 | Run `adb reverse tcp:5000 tcp:5000` in terminal. Ensure `BASE_URL=http://127.0.0.1:5000` in `local.properties`. Rebuild app. |
| **`CLEARTEXT_COMMUNICATION_NOT_PERMITTED`** | Android blocks non-HTTPS traffic | Use `127.0.0.1` or `localhost` (already whitelisted in `network_security_config.xml`). |
| **PostgreSQL connection refused in Flask** | PostgreSQL service is not running or wrong password | Ensure Docker container `shopflow-db` is running (`docker ps`) or Windows service `postgresql-x64-16` is started. Verify password in `.env`. |
| **Camera stream is black or fails to open** | Camera index is locked by another program | Close Windows Camera app, Zoom, or Teams. In Admin Panel (`http://localhost:5000/admin`), test with camera index `0` or `1`. |
| **Face check failed or low quality warning** | Lighting too dark, or face too far from camera | Stand closer to the camera (within 1 to 2 meters) with adequate lighting on the face. |
| **`No module named 'pgvector'`** | Virtual environment missing package | Run `.\venv\Scripts\activate` then `pip install pgvector psycopg2-binary`. |

---

## ⚡ 8. Quick Commands Reference

```powershell
# --- FLASK BACKEND ---
cd C:\Project\Autonomous-Retail-Checkout-System
.\venv\Scripts\activate
python app.py

# --- DOCKER DATABASE (IF USED) ---
docker run -d --name shopflow-db -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=shopflow pgvector/pgvector:pg16

# --- ADB REVERSE PORT FORWARDING (PHYSICAL PHONE) ---
adb devices
adb reverse tcp:5000 tcp:5000

# --- USEFUL URLS ---
# Surveillance Dashboard: http://localhost:5000/
# Admin Panel:            http://localhost:5000/admin
# Admin Login:            http://localhost:5000/admin/login
# Health Check Endpoint:  http://localhost:5000/api/health
```

---
*Documentation generated for ShopFlow Engineering Team.*
