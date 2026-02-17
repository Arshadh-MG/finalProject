# Wildlife Injury Detection — Vercel deployment

## Deploy on Vercel

1. Push this folder (`WildlifeInjuryDetection/`) to a Git repo.
2. In Vercel, import the repo and set **Root Directory** to `WildlifeInjuryDetection`.
3. Deploy (the project includes `vercel.json` that routes all requests to `app.py`).

## Deploy inference on Render (recommended)

Vercel can’t bundle/run the PyTorch + Ultralytics dependencies for this repo in a serverless function. Deploy the full app on Render (Docker) and let Vercel proxy `/image-upload` to it.

1. Create a Render account and click **New → Blueprint**.
2. Select your Git repo (root should be the `WildlifeInjuryDetection/` folder containing `render.yaml`).
3. Deploy. When it finishes, copy the Render service URL (example: `https://wildlife-injury-inference.onrender.com`).
4. Set the Vercel env var `INFERENCE_URL` to that Render URL and redeploy Vercel.

## Important Vercel constraints

- Vercel uses `requirements.txt` for dependency install. In this repo, `requirements.txt` is intentionally **minimal** so the app can deploy on Vercel without bundling PyTorch.
- Vercel Functions have a **read-only** filesystem (except a temp directory). This app auto-detects Vercel and writes uploads/logs to the temp directory.
- Runtime-generated annotated images cannot be written into `/static` on Vercel, so the image upload flow returns an inline `data:` URL for the annotated image.
- Video processing and live-camera streaming are **disabled on Vercel** in this repo (they are long-running and don’t fit typical serverless limits).
- To enable image inference on Vercel, deploy a separate inference server (same codebase with `requirements-full.txt`) and set `INFERENCE_URL` in Vercel Environment Variables (and optionally `INFERENCE_TOKEN`).

## Deploy with Vercel CLI (optional)

```powershell
cd WildlifeInjuryDetection
vercel
vercel --prod
```

## Local dev

```powershell
cd WildlifeInjuryDetection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-full.txt
python app.py
```
