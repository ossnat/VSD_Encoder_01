# claude_landmark_fits

Clean (35 px/deg) Schira fits against Gandalf landmarks, comparing a horizontal visual-field
stretch before Schira (rho) with alternatives. Written in a Claude session (Oct 2026).

Paths: the scripts were run in a cloud copy. Legacy files were read from
`/mnt/user-data/uploads/VSD_FM/legacy_code/...` = `VSD_FM/legacy_code/...` here, and
condition averages from `cond_avgs/` (this folder). Adjust `G` in landmarks.py and `D` in autolm.py.

| File | What it does |
|---|---|
| make_cond_avgs.py | per-condition trial averages from ProcessedData/gandalf/*.h5 -> cond_avgs/*.npz |
| landmarks.py | Schira (legacy convention theta/alpha, monopole double-sech), 35 ppd and legacy degree grids, 201118 landmarks |
| fit_models*.py | 201118: similarity / affine / pre-Schira stretch / fixation offset, with leave-one-out |
| run_100718*.py | same for 10/07/18 (thesis 6-pt file and xlsx 10-pt set) |
| joint.py | both days, shared a, alpha, (shared or per-day) rho |
| boromir.py | Boromir 241121 (4 dots), rho profile at fixed a, alpha |
| autolm.py, autofit.py, fig_auto.py | automatic landmarks (point peak, bar end peaks) from 5 shapes sessions; fits; figure |

Results summary: see the project notes `landmark-fits-201118.md`.
