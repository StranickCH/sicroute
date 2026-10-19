@echo off
rem Full recomputation of the paper, in order. Run from the Anaconda Prompt in this folder.
rem Every step can be restarted: finished downloads and finished training runs are skipped.
rem Training (step 5) takes the longest - leave it overnight.
set PYTHONUTF8=1
cd /d "%~dp0"
set W=12

python download_data.py era5 amsr2 masam2 nic carra || goto fail
python build_fields.py main carra nic --workers %W% || goto fail
python eval_products.py --workers %W% || goto fail
python eval_radar.py sigma amsr2 bias --workers %W% || goto fail
python train_agents.py drl radar_bc radar_privileged --workers %W% || goto fail
python train_agents.py radar_hybrid --seeds 5 --workers %W% || goto fail
python train_agents.py shaping --workers %W% || goto fail
python eval_radar.py nic noise --workers %W% || goto fail
python eval_replanning.py run --workers %W% || goto fail
python eval_commitment.py tune test --workers %W% || goto fail
python make_tables.py || goto fail
python make_figures.py || goto fail
echo Done. See results\summary.md and results\figures
pause
exit /b 0
:fail
echo Stopped with an error, see results\log.txt
pause
exit /b 1
