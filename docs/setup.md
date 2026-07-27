# Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Windows 11 | 22H2+ | |
| Docker Desktop | 4.x | WSL2 backend required |
| WSL2 | any | Enabled via Windows Features |
| Git | any | |
| VS Code | 1.85+ | Optional, for Dev Container |
| Remote - Containers extension | latest | Optional |

---

## Windows 11 Initial Setup

### Enable WSL2

Open PowerShell as Administrator:

```powershell
wsl --install
wsl --set-default-version 2
```

Restart your machine.

### Install Docker Desktop

1. Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. During installation, select **Use WSL2 instead of Hyper-V**
3. After install: **Docker Desktop → Settings → General → Use WSL2 backend** ✓
4. **Settings → Resources → WSL Integration** → enable for your distro

### Enable BuildKit (recommended)

In PowerShell:

```powershell
# User-level (persists across sessions)
[System.Environment]::SetEnvironmentVariable("DOCKER_BUILDKIT", "1", "User")
[System.Environment]::SetEnvironmentVariable("COMPOSE_DOCKER_CLI_BUILD", "1", "User")
```

Or prefix every build command:

```powershell
$env:DOCKER_BUILDKIT = "1"; docker compose build
```

---

## Project Setup

### 1. Clone

```powershell
git clone <repo-url> ocr-position-benchmark
cd ocr-position-benchmark
```

### 2. Create .env

```powershell
Copy-Item .env.example .env
notepad .env
```

Required variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LABEL_STUDIO_USERNAME` | Admin email | `admin@ocr.local` |
| `LABEL_STUDIO_PASSWORD` | Admin password | `changeme` |
| `LABEL_STUDIO_API_KEY` | API token (fill after first login) | empty |

### 3. Build images

```powershell
docker compose build
```

> **Expected time**: 10–20 minutes on first run (downloads ~2GB of packages).
> Subsequent builds are fast due to BuildKit layer caching.

### 4. Start services

```powershell
docker compose up -d
```

### 5. Verify health

```powershell
docker compose ps
```

Both services should show `healthy`:

```
NAME             STATUS              PORTS
label_studio     Up 2 minutes (healthy)   0.0.0.0:8080->8080/tcp
ocr_benchmark    Up 2 minutes (healthy)
```

---

## VS Code Dev Container

1. Open the `ocr-position-benchmark` folder in VS Code
2. When prompted **"Reopen in Container"** → click it
   (or: `Ctrl+Shift+P` → **Remote-Containers: Reopen in Container**)
3. VS Code attaches to the `ocr` container
4. All recommended extensions install automatically
5. Python interpreter, linter, and test runner are pre-configured

### Dev Container features

- Python 3.11 interpreter at `/usr/bin/python3.11`
- `ruff` formatter auto-runs on save
- `mypy` type checking via sidebar
- pytest test runner integrated (sidebar + `Ctrl+Shift+T`)
- Label Studio forwarded to port 8080

---

## Label Studio First-Time Setup

1. Open [http://localhost:8080](http://localhost:8080)
2. Click **Sign Up** (first visit) or log in with `.env` credentials
3. **Create Project** → name it (e.g., "OCR Benchmark")
4. **Labeling Setup** → choose **Object Detection with Bounding Boxes**
5. Configure the label interface XML (add a `<TextArea>` for transcription):

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="text" background="#00aeff"/>
  </RectangleLabels>
  <TextArea name="transcription" toName="image"
            editable="true" perRegion="true" required="false"
            maxSubmissions="1" rows="1" placeholder="Transcription"/>
</View>
```

6. **Import** your images (drag-and-drop or via Local Files)
7. Annotate each image by drawing bounding boxes + typing the text
8. **Export** → select **JSON** → save to `annotations/export.json`

---

## Label Studio API Key Setup

After first login:

1. Click your avatar (top-right) → **Account & Settings**
2. Copy the **Access Token**
3. Add it to `.env`:
   ```
   LABEL_STUDIO_API_KEY=<your-token-here>
   ```

---

## Running the Pipeline

```powershell
# 1. Run OCR
docker compose exec ocr python scripts/run_ocr.py --engine paddle --images /workspace/images
docker compose exec ocr python scripts/run_ocr.py --engine tesseract --images /workspace/images

# 2. Score
docker compose exec ocr python scripts/score.py

# 3. Visualize
docker compose exec ocr python scripts/visualize.py

# 4. View reports
ls reports/
```

---

## Stopping and Restarting

```powershell
# Stop (keep data)
docker compose stop

# Restart
docker compose start

# Full reset (WARNING: deletes all volumes and data)
docker compose down -v
```
