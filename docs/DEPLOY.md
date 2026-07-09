# Deploy Rice Leaf Detection (Free Tier)

Zero-cost stack:

| Layer | Service |
|-------|---------|
| UI | Streamlit Community Cloud |
| API | GCP Cloud Run (always-free tier) |
| Models | Hugging Face Hub (public repo) |
| CI/CD | GitHub Actions |

## Prerequisites

1. Public GitHub repo: [github.com/jegadeesh17/RiceLeafDetection](https://github.com/jegadeesh17/RiceLeafDetection)
2. [Hugging Face](https://huggingface.co) account
3. [Google Cloud](https://cloud.google.com) account (card required; stays free within Cloud Run limits)
4. Trained model at `models/ai_system_rice_leaf_final.keras`

## Step 1 — Upload model to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login

python scripts/upload_model_to_hf.py --repo-id YOUR_USERNAME/rice-leaf-disease-model
```

Use a **public** model repo so Cloud Run and Streamlit can download without tokens.

## Step 2 — Deploy Streamlit UI (free, no GCP)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. New app → select this repo
4. Main file: `app/app.py`
5. Add secrets (Settings → Secrets):

```toml
HF_MODEL_REPO = "YOUR_USERNAME/rice-leaf-disease-model"
```

6. Deploy. First load downloads the model (~30 MB) and may take 1–2 minutes.

## Step 3 — One-time GCP setup for Cloud Run API

```bash
# Install gcloud CLI, then:
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud services enable run.googleapis.com artifactregistry.googleapis.com

gcloud artifacts repositories create ml-apis \
  --repository-format=docker \
  --location=asia-south1

# Service account for GitHub Actions
gcloud iam service-accounts create github-deployer

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud iam service-accounts keys create gcp-key.json \
  --iam-account=github-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## Step 4 — GitHub secrets for API deploy

In repo **Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_SA_KEY` | Contents of `gcp-key.json` |
| `HF_MODEL_REPO` | `YOUR_USERNAME/rice-leaf-disease-model` |

## Step 5 — Deploy API to Cloud Run

1. GitHub → **Actions** → **Deploy API to Cloud Run** → **Run workflow**
2. When finished, note the service URL from the workflow log
3. Test:

```bash
curl https://YOUR-SERVICE-xxx.run.app/health
curl -X POST https://YOUR-SERVICE-xxx.run.app/predict -F "file=@leaf.jpg"
```

## Local Docker test (optional)

```bash
docker build -t rice-leaf-api .
docker run -p 8080:8080 -e HF_MODEL_REPO=YOUR_USERNAME/rice-leaf-disease-model rice-leaf-api
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
