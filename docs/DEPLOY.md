# Deploy Rice Leaf Detection (Free Tier)

**Status: deployed and verified** (July 2026)

| Surface | Status | URL / entry |
|---------|--------|-------------|
| Streamlit (local) | Verified | `streamlit run app/app.py` |
| Streamlit Cloud | Verified | `app/app.py` on [share.streamlit.io](https://share.streamlit.io) |
| Cloud Run API | Verified | https://rice-leaf-api-5obmkzpuaa-el.a.run.app |
| Hugging Face model | Verified | https://huggingface.co/jegadeesh17/rice-leaf-disease-model |
| GCP project | `ml-portfolio-501915` | region `asia-south1` |

**Endpoints**

- API health: https://rice-leaf-api-5obmkzpuaa-el.a.run.app/health
- Swagger: https://rice-leaf-api-5obmkzpuaa-el.a.run.app/docs
- Browser UI (after latest deploy): https://rice-leaf-api-5obmkzpuaa-el.a.run.app/

**Note:** Streamlit Cloud may load the model from **Git LFS** in the repo (file exists at `models/` after clone). Cloud Run loads from **Hugging Face** at container startup (`HF_MODEL_REPO`).

---

Zero-cost stack:


| Layer | Service |
|-------|---------|
| UI | Streamlit Community Cloud |
| API | GCP Cloud Run (always-free tier) |
| Models | Hugging Face Hub (public repo) |
| CI/CD | GitHub Actions |

## Prerequisites

1. Public GitHub repo: [github.com/jegadeesh17/AI-powered-rice-leaf-detection-system](https://github.com/jegadeesh17/AI-powered-rice-leaf-detection-system)
2. [Hugging Face](https://huggingface.co) account
3. [Google Cloud](https://cloud.google.com) account (card required; stays free within Cloud Run limits)
4. Trained model at `models/ai_system_rice_leaf_final.keras`

## Step 1 — Upload model to Hugging Face

```bash
pip install huggingface_hub
hf auth login

python scripts/upload_model_to_hf.py --repo-id jegadeesh17/rice-leaf-disease-model
```

Use a **public** model repo so Cloud Run and Streamlit can download without tokens.

## Step 2 — Deploy Streamlit UI (free, no GCP)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. New app → select this repo
4. Main file: `app/app.py`
5. Add secrets (Settings → Secrets):

```toml
HF_MODEL_REPO = "jegadeesh17/rice-leaf-disease-model"
```

6. Deploy. First load downloads the model (~30 MB) and may take 1–2 minutes.

## Step 3 — One-time GCP setup for Cloud Run API

```bash
# Install gcloud CLI, then:
gcloud auth login
gcloud config set project ml-portfolio-501915

gcloud services enable run.googleapis.com artifactregistry.googleapis.com

gcloud artifacts repositories create ml-apis \
  --repository-format=docker \
  --location=asia-south1

# Service account for GitHub Actions
gcloud iam service-accounts create github-deployer

gcloud projects add-iam-policy-binding ml-portfolio-501915 \
  --member="serviceAccount:github-deployer@ml-portfolio-501915.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding ml-portfolio-501915 \
  --member="serviceAccount:github-deployer@ml-portfolio-501915.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding ml-portfolio-501915 \
  --member="serviceAccount:github-deployer@ml-portfolio-501915.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud iam service-accounts keys create gcp-key.json \
  --iam-account=github-deployer@ml-portfolio-501915.iam.gserviceaccount.com
```

## Step 4 — GitHub secrets for API deploy

In repo **Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | `ml-portfolio-501915` |
| `GCP_SA_KEY` | Contents of `gcp-key.json` |
| `HF_MODEL_REPO` | `jegadeesh17/rice-leaf-disease-model` |

## Step 5 — Deploy API to Cloud Run

1. GitHub → **Actions** → **Deploy API to Cloud Run** → **Run workflow**
2. When finished, note the service URL from the workflow log
3. Test:

```bash
curl https://rice-leaf-api-5obmkzpuaa-el.a.run.app/health
curl -X POST https://rice-leaf-api-5obmkzpuaa-el.a.run.app/predict -F "file=@leaf.jpg"
```

## Local Docker test (optional)

```bash
docker build -t rice-leaf-api .
docker run -p 8080:8080 -e HF_MODEL_REPO=jegadeesh17/rice-leaf-disease-model rice-leaf-api
```

## Interview talking points

- Model artifacts decoupled from container image via Hugging Face Hub
- API on Cloud Run scales to zero (no cost when idle)
- Streamlit Cloud for demo UI without managing servers
- GitHub Actions for repeatable deploys

## Cost notes

- Streamlit Cloud: free for public repos
- Hugging Face Hub: free for public models
- Cloud Run free tier: ~2M requests/month — more than enough for portfolio demos

## Two cloud surfaces (recommended)

| Surface | URL / entry | Purpose |
|---------|-------------|---------|
| **Streamlit dashboard** | Streamlit Cloud → `app/app.py` | Rich demo: Grad-CAM, treatments, telemetry |
| **API + browser UI** | Cloud Run `/` | REST API with built-in upload page; `/docs` for Swagger |

### Built-in web UI on Cloud Run

After deploying the API, open:

```
https://rice-leaf-api-5obmkzpuaa-el.a.run.app/
```

Upload an image in the browser — it calls `POST /predict` on the same service. No separate Streamlit API client needed.

**Local full app:** `streamlit run app/app.py` (same rich UI, local or LFS model).
