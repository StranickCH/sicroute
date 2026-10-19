"""Train and evaluate the PPO agents. Every run writes results/runs/<id>/eval.csv; finished
runs are skipped, so the script can be stopped and started again.

  drl               PPO without radar ("pure" and "hybrid" with the planner hint), one per product and seed
  radar_hybrid      hybrid PPO with a radar spot, every (R, sigma) of the grid
  radar_bc          the same, initialised by behaviour cloning of the planner
  radar_privileged  agent imitating a fusion planner that knows the true sigma (BC + DAgger + PPO);
                    needs tables/sigma_bg.json, i.e. run "eval_radar.py sigma" first
  shaping           ablation of the reward shaping (gamma * phi(s') - phi(s) and no shaping) for the
                    radar-aware agent with R = 4 and hybrid PPO on ERA5; into results/runs_shaping

Takes long: each experiment is several hours on a 12-core desktop.
The paper used --seeds 5 for radar_hybrid; the other experiments use the seeds in config.py.

  python train_agents.py radar_privileged --workers 12
  python train_agents.py drl --smoke          # a few thousand steps per run, to check the setup
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from sicroute.agents import job_id, make_jobs, run_job
from sicroute.config import CFG, RESULTS_DIR, RUNS_DIR, log

EXPERIMENTS = ["drl", "radar_hybrid", "radar_bc", "radar_privileged", "shaping"]


def run_pool(jobs, workers):
    todo = [j for j in jobs if not (Path(j.get("out_root", RUNS_DIR)) / job_id(j) / "eval.csv").exists()]
    log(f"{len(jobs)} runs, {len(todo)} left, {workers} processes")
    t0, n, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_job, j): j for j in todo}
        for f in as_completed(futs):
            n += 1
            try:
                f.result()
            except Exception as e:          # keep the other runs going, report at the end
                failed += 1
                log(f"  FAILED {job_id(futs[f])}: {e!r}")
            el = time.time() - t0
            log(f"  {n}/{len(todo)} {job_id(futs[f])}; {el / 3600:.2f} h, about {el / n * (len(todo) - n) / 3600:.2f} h left")
    return failed


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("experiments", nargs="+", choices=EXPERIMENTS)
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    a.add_argument("--seeds", type=int, help="override the number of seeds")
    a.add_argument("--smoke", action="store_true", help="short runs on seed 0 into results/smoke")
    a = a.parse_args()
    jobs = []
    for e in a.experiments:
        jobs += make_jobs(CFG, e, seeds=1 if a.smoke else a.seeds)
    if a.smoke:
        seen = set()
        small = []
        for j in sorted(jobs, key=lambda j: -j.get("R", 0)):   # one run per variant, with the radar on
            k = (j["kind"], j["variant"], j.get("shaping", "unit"))
            if k in seen:
                continue
            seen.add(k)
            j.update(steps=4096, n_demo=2000, dagger_steps=1000, max_eval_scen=4, out_root=str(RESULTS_DIR / "smoke"))
            small.append(j)
        jobs = small
    failed = run_pool(jobs, a.workers)
    if failed:
        raise SystemExit(f"{failed} runs failed, see {RESULTS_DIR / 'log.txt'}")


if __name__ == "__main__":
    main()
