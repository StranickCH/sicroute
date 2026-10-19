# sicroute

Code for the paper *Sea ice data uncertainty in voyage planning in the Barents and Kara Seas:
the cost of product error, on-board radar and the limits of reinforcement learning*
(B. Danilov, J. Dong, A. Said Ben Salah).

Two routes from the Kola Bay (to the approaches of the Gulf of Ob and to the Prirazlomnaya
platform), April-December 2022-2024. A ship plans its route on one sea ice concentration (SIC)
product and sails through another one, which plays the role of the truth. We measure the regret
against a planner that sees the truth, then add an on-board radar and compare classical data
fusion with DRL agents.

## Data

All data are open. Only the Copernicus CDS needs an account (free): put your key into
`~/.cdsapirc` (on Windows `%USERPROFILE%\.cdsapirc`) and accept the dataset licences on the CDS
website. The key is never stored in this repository.

| Product | Source | Used as |
|---|---|---|
| ERA5 sea-ice cover | Copernicus CDS, doi:10.24381/cds.adbb2d47 | product, radar background |
| AMSR2 ASI 6.25 km | University of Bremen, data.seaice.uni-bremen.de | product, radar truth |
| MASAM2 v2 | NSIDC G10005 | product |
| CARRA-East | Copernicus CDS, doi:10.24381/cds.713858f6 | fourth product, route R2 only |
| U.S. NIC weekly charts (SIGRID-3) | NSIDC G10013, doi:10.7265/4b7s-rn93 | independent truth |

`download_data.py` puts everything into `data/` (several GB; CARRA is the largest part). Another location can be given with
the environment variable `SICROUTE_DATA`, the results folder with `SICROUTE_RESULTS`.

## Installation

Python 3.10 or newer.

    conda install -c conda-forge pyhdf
    pip install -r requirements.txt

## Running

`run_all.bat` runs everything in order. Step by step:

    python download_data.py era5 amsr2 masam2 nic carra
    python build_fields.py main carra nic --workers 12
    python eval_products.py --workers 12          # products, planner, CARRA, NIC, ship model
    python eval_radar.py sigma amsr2 bias --workers 12
    python train_agents.py drl radar_bc radar_privileged --workers 12
    python train_agents.py radar_hybrid --seeds 5 --workers 12
    python train_agents.py shaping --workers 12   # reward shaping ablation, results/runs_shaping
    python eval_radar.py nic noise --workers 12   # needs the trained radar-aware agents
    python eval_replanning.py run --workers 12    # course changes and noise sensitivity of the radar methods
    python eval_commitment.py tune test --workers 12   # fusion that keeps its route: test of the mechanism
    python make_tables.py
    python make_figures.py

Every step skips what is already done, so it can be interrupted and started again. Training is
the slow part (one night on a 12-core desktop). `python train_agents.py drl --smoke` runs a few
short jobs to check the setup first.

All settings are in `sicroute/config.py`. To change some of them without editing the file, write
the keys into a JSON file and set `SICROUTE_CONFIG`, for example a quick run on one year:

    {"years": [2024], "train_years": [2024], "test_years": [2024], "depart_step_days_test": 20}

## Layout

    sicroute/config.py     settings, paths, grid
    sicroute/fields.py     download and regridding of ERA5, AMSR2, MASAM2; land mask; QA
    sicroute/carra.py      CARRA-East
    sicroute/nic.py        NIC charts: SIGRID-3 codes, rasterisation
    sicroute/world.py      graph, ship model, Dijkstra planner
    sicroute/env.py        voyage environment (also the radar)
    sicroute/policies.py   planner, radar fusion methods, background error
    sicroute/agents.py     PPO, BC + PPO, privileged expert + DAgger + PPO; training jobs
    sicroute/stats.py      bootstrap, Holm, regret

Fields go to `results/fields`, per-voyage results to `results/tables`, trained agents to
`results/runs/<run>/` (policy, training log, evaluation), figures to `results/figures`, a log of
all steps to `results/log.txt`.

## Result tables

`paper_tables.zip` holds the `results/tables` folder of the full run behind the paper: the
per-voyage results of every method (planner, radar baselines, trained agents) and the summary
tables (see the map below). Unzip it into `results/` to check the numbers without recomputing.
`make_figures.py` then rebuilds the figures whose inputs are all in the archive; the maps
(Figs. 2 and 6) also need the ice fields from `build_fields.py`, and the figures of the trained
agents need `results/runs` from `train_agents.py`.

## Where the numbers of the paper come from

| Paper (S = supplementary material) | File in `results/tables` (or `results/figures`) |
|---|---|
| Fig. 1 (scheme of the study) | `fig0_method_scheme.png` |
| Fig. 2 | `fig1_products_by_regime.png` |
| Table S1, Fig. 3 | `table2_agreement.csv`, `fig2_iiee_timeseries.png` |
| Table 2 | `table2b_nic_agreement.csv` |
| Table 3, Fig. 4 | `table3_planner_cross_eval.csv`, `fig3_planner_regret.png` |
| Fig. 5, Table S2 (planner, truth = NIC) | `table3b_nic_planner.csv`, `fig_products_vs_nic.png` |
| Fig. 6 (one voyage, routes on each product) | `fig_route_example.png` (re-runs the voyage, checks it against `nic_planner.csv`) |
| CARRA on route R2 (S4) | `carra_planner_R2_regret.csv`, `carra_agreement.csv` |
| Ship model sensitivity (S5) | `sensitivity_ship_model.csv` |
| Table 4, Fig. S1 | `table4_drl_vs_planner.csv`, `drl_hybrid_vs_pure.csv`, `fig4_drl_vs_planner.png` |
| Reward shaping ablation, Table S3 | `shaping_ablation.csv`, `shaping_by_regime.csv` |
| Fig. 7, Table S4 | `table5_radar_amsr2.csv`, `agent_vs_fusion_amsr2.csv`, `sigma_hat_accuracy.csv`, `fig5_radar_amsr2.png` |
| Fig. 8, Tables S5–S7 (radar error models, course changes) | `radar_noise_regret.csv`, `agent_vs_fusion_noise.csv`, `replanning_summary.csv` (per voyage: `replanning.csv`), `fig7_error_models.png` |
| Fusion with a commitment rule, Table S8 | `commitment_summary.csv` (tuning on 2023: `commit_tune.csv`) |
| Fig. 9, Table S9 | `table6_radar_nic.csv`, `agent_vs_fusion_nic.csv`, `radar_nic_by_regime.csv`, `fig6_radar_nic.png` |
| Background bias, bias-aware fusion (S11) | `bias_diag_by_regime.csv`, `radar_bias_regret.csv`, `fusion_bias_vs.csv` |
| Table S10, Fig. S2 | `table5_radar_amsr2.csv` (rows `agent_hybrid`, `agent_bc`), `bc_radar_vs_no_radar.csv`, `figA1_training_curves.png` |

`results/summary.md` collects all tables in one file.

## Notes

- Regret is measured against the better of two planners that see the truth: `oracle` follows
  today's map and replans every step, `foresight` knows the whole future of the field and
  stops on entering the 25 km goal circle, where the voyage ends (`World.plan_foresight`,
  dynamic programming over (cell, arrival time) states with a 15-minute step, set by
  `foresight_step_h`). The planners aim at the goal cell itself, which costs them about
  1-2 % of regret even in open water.
- Confidence intervals over scenarios resample 15-day blocks of departures, because voyages
  that leave a few days apart meet nearly the same ice.

- The reward uses the shaping term phi(s') - phi(s) with factor 1 instead of the usual
  gamma * phi(s') - phi(s) (Ng et al., 1999); this is our modification. `train_agents.py shaping`
  retrains the radar-aware agent (R = 4) and hybrid PPO on ERA5 with the standard form and without
  shaping, so the effect of this choice can be checked.
- Random seeds are fixed (training: the run seed; radar noise in the planner baselines: 777), so
  the same data give the same numbers. PyTorch on a different CPU or version may differ in the
  last digits of the trained weights.
- The NIC and CARRA steps stop with an error if the regridded field correlates poorly with the
  other products; this is a check of the georeferencing, not of the data quality.
- In the code the agent of the paper called "v4" is `privileged` and "v3" is `bc`.

## Licence

MIT, see `LICENSE`. Please cite the paper if you use the code (`CITATION.cff`).
