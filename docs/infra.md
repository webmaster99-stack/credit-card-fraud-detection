# Infrastructure notes: free-tier limits

Recorded **2026-09-25** from each vendor's official pages. Limits change; re-check before relying on
any of them. Everything must stay rebuildable from git + DVC in case a free tier disappears.

| Service | Used for | Free-tier limits (as recorded) | Source |
| --- | --- | --- | --- |
| DagsHub | Git mirror, DVC remote, hosted MLflow | 20 GB DagsHub Storage; unlimited public repos; unlimited private repos for non-commercial use; up to 2 collaborators in private projects; **experiment tracking: up to 100 tracked experiments in private repos, unlimited in public repos** | dagshub.com/pricing |
| Hugging Face | Model repo (model card + pipeline bundle), Demo Space | CPU Basic hardware: 2 vCPU, 16 GB RAM, 50 GB non-persistent disk, free-tier Spaces sleep when idle. **Creating a Gradio or Docker Space requires a paid plan (PRO/Team/Enterprise).** Static Spaces are free. Free personal accounts can host up to 2 Gradio Spaces on ZeroGPU. | huggingface.co/docs/hub/spaces-overview |
| Render | FastAPI backend (Phase 5) | 750 free instance hours per workspace per month; spins down after 15 min idle, ~1 min to wake; single instance only; no persistent disk; no SSH/shell; no one-off jobs; may be suspended for unusually high outbound traffic | render.com/docs/free |
| Vercel | Next.js frontend (Phase 5) | Hobby: 100 deployments/day; 45 min max build; 100 MB source upload via CLI; 1 concurrent build; runtime logs kept 1 hour; Hobby is for non-commercial use. Monthly usage allotments (bandwidth, invocations) are on the fair-use page, not recorded here. | vercel.com/docs/limits |
| Neon | Postgres for prediction log + card history (Phase 5) | 0.5 GB storage per project; 100 CU-hours per project; 100 projects; 10 branches per project; scale-to-zero after 5 min (cannot be turned off); exceeding limits suspends compute until the next cycle, no data deleted | neon.com/pricing |

## Project resources

| Resource | Location |
| --- | --- |
| Code (source of truth, CI) | https://github.com/webmaster99-stack/credit-card-fraud-detection |
| DagsHub (git mirror, DVC remote, MLflow) | https://dagshub.com/webmaster99-stack/credit-card-fraud-detection |
| DVC remote | `https://dagshub.com/webmaster99-stack/credit-card-fraud-detection.dvc` (credentials in git-ignored `.dvc/config.local`) |
| MLflow tracking URI | `https://dagshub.com/webmaster99-stack/credit-card-fraud-detection.mlflow` (credentials in git-ignored `.env`) |
| HF Hub model repo | https://huggingface.co/ilian-hadzhidimitrov/fraud-classifier |
| HF Space (Gradio demo) | https://huggingface.co/spaces/ilian-hadzhidimitrov/fraud-classifier-demo (free ZeroGPU slot) |
| GitHub Actions secrets | `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` |

## Decisions made on these limits

- **Demo host: Hugging Face Spaces on a free ZeroGPU slot (owner's choice, 2026-09-25).** Creating a
  regular Gradio Space needs a paid HF plan, so the demo uses one of the 2 free ZeroGPU Gradio slots.
  ZeroGPU is a GPU runtime, but the project rule is that every script runs on CPU. Phase 4 must keep
  the demo free of GPU-specific code and verify that a CPU-only model serves correctly on a ZeroGPU
  Space (including whether the `spaces` decorator is needed) before relying on it.

### ZeroGPU verification (2026-09-28, Phase 4)

- **The Space runs on `zero-a10g` and serves the CPU-only model correctly**: single scoring, batch CSV and the
  example buttons were exercised through the Space's public API and match local scores (template fraud row 89.2%).
- **ZeroGPU ignores `python_version: "3.11"`** and builds on Python 3.10, where the pinned `numpy==2.4.6` does not
  exist (first build failed with `BUILD_ERROR`). `demo/README.md` sets `python_version: "3.12.12"`, which built and
  ran. The project itself is developed on Python 3.11, so the pickled pipeline is loaded under a newer minor version;
  scores matched locally, but re-check them after any pin change.
- The Space installs its own pins from `demo/requirements.txt` (checked against `uv.lock` by
  `tests/test_space_requirements.py`) and vendors `src/fraud` next to `app.py`; it needs no DVC or MLflow credentials.
- `demo/app.py` defines an unused `@spaces.GPU` function so the Space passes ZeroGPU's startup check. It works with
  the probe present; whether the probe is actually required was **not** tested without it.
- `HF_TOKEN` is not in `.env`; the export and Space uploads used the cached `huggingface-cli` login.

## Open issues

- **DagsHub experiment cap.** If the repo is private, only 100 tracked experiments are free. A
  public repo has no cap. Sweeps with Optuna in Phase 3 should log one parent run with nested trials
  or be mindful of this cap.
- Vercel monthly bandwidth/invocation allotments and Render outbound bandwidth allotments were not
  stated on the pages consulted. Look them up before Phase 5.

## Phase 5 setup (owner action required)

The API (`api/`) and its tests are built and pass against a real Postgres (verified locally via
`docker-compose.yml`'s `db` service - see ADR 0005). Three accounts still need to be created by hand
before the service is actually live; none of this can be done from the repo alone.

1. **Neon**: create a project, copy its connection string into `DATABASE_URL` (a GitHub secret for
   CI/deploy, and a Render environment variable for the running service). `api/db.py`'s
   `init_schema` creates the `predictions` table itself on first startup - no separate migration
   step.
2. **Render**: new Web Service, "Docker" runtime, this GitHub repo, Dockerfile path `api/Dockerfile`,
   root directory `.`. Environment variables: `DATABASE_URL` (from Neon), `API_KEY` (any long random
   string - the frontend needs the same value), `ALLOWED_ORIGINS` (the Vercel URL once it exists),
   `MODEL_REVISION=champion`. **Turn off Render's auto-deploy on push** - `.github/workflows/ci.yml`'s
   `deploy-api` job deploys deliberately, only on a `v*` tag, once the image has built and tests have
   passed. Copy the service's Deploy Hook URL (Settings -> Deploy Hook) into the GitHub secret
   `RENDER_DEPLOY_HOOK_URL`.
3. Before the first deploy, export and push the champion bundle so the branch the API downloads from
   actually exists: `uv run python -m fraud.serving.export --alias champion --push`.
4. Free tier: the service sleeps after 15 min idle and takes about a minute to wake on the first
   request after that (recorded in the table above). `web/components/ApiStatusBanner.tsx` already
   polls `/health` and shows a "waking up" banner while it isn't `ok`.
5. **Vercel**: import this GitHub repo, set the project root to `web/`. Environment variables (set
   as server-only, i.e. not prefixed `NEXT_PUBLIC_`): `FRAUD_API_URL` (the Render service's URL),
   `FRAUD_API_KEY` (the same value as the API's `API_KEY`). The app is built and passes `next build`
   locally (`web/package.json`); no code changes should be needed to deploy it as-is.

**Not yet done, blocked on this machine's C: drive being full (0 bytes free, 2026-09-29):** building
`api/Dockerfile` end-to-end with `docker build`. The dependency resolution ran correctly (confirmed:
uv resolved and started downloading all ~260 packages without error) before Docker Desktop's own
storage went read-only and crashed. Re-run `docker compose up --build` once there's disk space, to
confirm the image actually builds and serves before relying on Render to build the same Dockerfile.
