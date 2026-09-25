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

## Open issues

- **Hugging Face Spaces vs. the "free demo host" decision.** CLAUDE.md settles on a Gradio demo on
  HF Spaces. As of the date above, creating a Gradio Space needs a paid HF plan. Options to decide
  with the owner: (a) pay for PRO, (b) use one of the 2 free ZeroGPU Gradio slots (a GPU-oriented
  runtime for a CPU-only model; needs verification), (c) use a different free host for the Gradio
  app. Not decided; do not change the decision without the owner.
- **DagsHub experiment cap.** If the repo is private, only 100 tracked experiments are free. A
  public repo has no cap. Sweeps with Optuna in Phase 3 should log one parent run with nested trials
  or be mindful of this cap.
- Vercel monthly bandwidth/invocation allotments and Render outbound bandwidth allotments were not
  stated on the pages consulted. Look them up before Phase 5.
