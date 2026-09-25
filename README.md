# PocketSmart AI — Complete Project

A full working FastAPI + Jinja2 implementation of the supplied PocketSmart AI report.

## Included

- JWT authentication using HttpOnly cookies and `/token`
- SQLite persistence for users, sessions, and recommendation history
- 30-minute idle session cleanup
- Home, Party, and Jewelry planners
- Optional outfit image for Jewelry Planner
- Gemini integration using the current `google-genai` SDK
- Demo fallback mode if no API key is configured
- Platform search links
- Swagger `/docs` and ReDoc `/redoc`
- Automated pytest tests

## VS Code setup — Windows

Open this folder in VS Code, then open **Terminal > New Terminal**.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemini-2.5-flash
SECRET_KEY=replace_this_with_a_long_random_secret
```

`GOOGLE_API_KEY` is optional. Without it, all pages still work in deterministic demo mode.

Generate a secret:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Run:

```powershell
uvicorn app:app --reload
```

Open:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/redoc

If PowerShell blocks activation, use a Command Prompt terminal:

```bat
.venv\Scripts\activate.bat
```

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload
```

## Test

```bash
pytest
```

Manual test flow:

1. Register
2. Login
3. Generate Home plan
4. Generate Party plan
5. Generate Jewelry plan without an image
6. Generate Jewelry plan with JPG/PNG/WebP
7. Open History
8. Check `/docs`

## Important

The app generates estimated recommendations and search links; it does not claim real-time marketplace prices.
For production: use HTTPS, set cookie `secure=True`, add rate limiting, and use a managed database.
