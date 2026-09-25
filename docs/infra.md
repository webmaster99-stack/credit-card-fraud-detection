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

## Open issues

- **DagsHub experiment cap.** If the repo is private, only 100 tracked experiments are free. A
  public repo has no cap. Sweeps with Optuna in Phase 3 should log one parent run with nested trials
  or be mindful of this cap.
- Vercel monthly bandwidth/invocation allotments and Render outbound bandwidth allotments were not
  stated on the pages consulted. Look them up before Phase 5.
