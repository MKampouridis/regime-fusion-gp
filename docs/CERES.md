# Running on CERES

CERES (University of Essex) is a Rocks 7 cluster (CentOS 7.4, glibc 2.17) whose jobs are managed by Open Grid Scheduler, a version of **Sun Grid Engine**: jobs are submitted with `qsub`, and job arrays number their elements with `SGE_TASK_ID`, from 1. It is reached with `ssh <user>@ceres.essex.ac.uk` from the campus network or the university VPN, with multi-factor authentication. The home folder (`/home/<user>`, on shared storage) is visible from every node.

A campaign is a list of numbered tasks ([`ENGINE.md`](ENGINE.md), Campaigns); each element of a job array does one of them. Tasks can be repeated safely, so a failed or stopped element is just submitted again.

## 1. The cluster's rules

Checked on 29 October 2026 from Michael's terminal and the cluster's guide, [hpc.essex.ac.uk](https://hpc.essex.ac.uk) (Grid Engine, Python, GPU and file pages):

- **Queues.** `all.q` is for batch work and job arrays (1,751 slots, one per core). `allhi.q` is only for up to 30 jobs when `all.q` is busy, `interactive.q` for interactive sessions (`qrsh`), and `gpu.q` (24 GPUs: 16 GTX 1080 Ti and 8 RTX 2080) for batch jobs that use a GPU.
- **Limits.** A user may have 260 jobs. A job array counts as one, but must limit how many of its elements run at once with `-tc`: at most 60, and no more than about 250 across all of a user's arrays. An array may have up to 75,000 elements. `all.q` has no run-time or memory limit, but its jobs are suspended while `allhi.q` is very busy, and resume by themselves.
- **Scripts** must say they use bash (`-S /bin/bash`), and a job that uses several cores must ask for them (`-pe smp N`); the study's tasks use one core each.
- **Python.** The system's Pythons (3.6, and 3.9 on the compute nodes) are too old for this code (3.11 or later). CERES recommends conda environments, built in an interactive session on a compute node, not on the login node. The compute nodes can download packages (conda-forge answered from the login node too).
- **Disk.** No quotas, but space should be used sensibly. Each node's `/tmp` is fast local scratch; the per-job folder `$TMPDIR` is removed when the job ends.

The commands below run on the **login node** (where `ssh` lands), with two exceptions: the environment's set-up (§3), which CERES wants on a compute node, in an interactive session, and the data copy (§4), which runs on the Mac. Job arrays are submitted from the login node and run on the compute nodes by themselves.

## 2. The code

The repository is public, so CERES clones it directly (login node):

```bash
git clone https://github.com/MKampouridis/regime-fusion-gp.git ~/regime-fusion-gp
cd ~/regime-fusion-gp && git pull        # later, to bring in changes
```

## 3. The environment

```bash
screen                                                 # optional: keeps the session alive if the connection drops
qrsh -q interactive.q                                  # an interactive session on a compute node
cd ~/regime-fusion-gp && bash scripts/ceres/setup.sh
exit                                                   # back to the login node
```

The script installs Miniforge in `~/miniforge3`, creates the environment `rfgp` from conda-forge (Python 3.12 and the study's packages, PyTorch if it installs), installs the package and runs the tests (10–20 minutes, about 3 GB in the home folder). conda-forge still builds for glibc 2.17, and conda installs only builds that work with it, so versions may be older than on the Mac; the tests show whether they behave the same. If it is cut short, run it again: it redoes only what is missing. In `screen`, Ctrl-A then D leaves the session running, and `screen -r` returns to it.

Done on 29 October 2026: Python 3.12.16, numpy 2.5.4, pandas 3.1.0, numba 0.68.0, SciPy 1.18.1, scikit-learn 1.9.1, LightGBM 4.7.0 and PyTorch 2.13.0 (CPU), the Mac's versions but for patch releases, and all 366 tests pass (in 92 seconds, against 48 on the Mac). exchange-calendars is older (4.8, against 4.13), but only the data steps, which run on the Mac, use it.

## 4. The data

Runs read the prices, the processed indicators (FinBERT's included) and the regimes, about 80 MB. From the Mac, once the data steps have run (each `rsync` asks for the password and the second factor):

```bash
ssh <user>@ceres.essex.ac.uk mkdir -p rfgp-data/interim
rsync -av ~/rfgp-data/interim/prices ~/rfgp-data/interim/news <user>@ceres.essex.ac.uk:rfgp-data/interim/
rsync -av ~/rfgp-data/processed <user>@ceres.essex.ac.uk:rfgp-data/
```

Raw licensed data (FNSPID) are not needed on CERES: FinBERT scores headlines on the Mac, and only the daily indicators travel.

## 5. A campaign

These commands run on the login node: `qsub` refuses to run in an interactive session (`qrsh`), so leave it with `exit` first (`hostname` shows `compute-…` inside one). The plans can be made on either, since the home folder is shared.

The main campaign (the 50 main stocks, test years 2019–2025, every method of `PROTOCOL.md` §2) has three plans: the exit settings (350 tasks, one per stock-year), the runs of every method but ML-TFT (6,300 tasks: 18 methods × 350 stock-years, each task all 50 runs of one method on one stock-year), and ML-TFT, which is pooled across stocks (70 tasks: 7 test years × 10 runs, one run per task). Planned on a copy of the data on 1 November 2026 to check the counts.

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rfgp
unset PYTHONPATH && export RFGP_DATA_DIR=~/rfgp-data
cd ~/regime-fusion-gp && mkdir -p logs
E="python -m regime_fusion.engine"
$E tasks --campaign main --kind exits
$E tasks --campaign main --kind runs --rungs A0 A1 A2 A3 A4 A5 A5-dalex GP-FA GP-SA GP-TA \
    RB-BH RB-TS RB-MA RB-TSMOM ML-LR ML-LGBM ML-RS-LGBM ML-GATED
$E tasks --campaign main --kind runs --plan tft --rungs ML-TFT --block 1

# First, two elements as a test (each task prints the qsub line of its plan):
qsub -t 1-2 -tc 2 -v CAMPAIGN=main,PLAN=exits scripts/ceres/tasks.sge
qstat -u $USER                     # what is queued (qw) or running (r)
cat logs/rfgp.o*                   # once they finish: no errors? how long did they take?

# Then the rest. The runs need their stock-year's exit settings, so they wait for the exits job
# (its number is printed by qsub, as "Your job-array 12345.3-350:1 ..."):
qsub -t 3-350 -tc 60 -v CAMPAIGN=main,PLAN=exits scripts/ceres/tasks.sge
qsub -hold_jid <exits job> -t 1-6300 -tc 60 -v CAMPAIGN=main,PLAN=runs scripts/ceres/tasks.sge
qsub -hold_jid <exits job> -t 1-70 -tc 60 -v CAMPAIGN=main,PLAN=tft scripts/ceres/tasks.sge
$E status --campaign main          # what is finished
```

`-hold_jid` must name the job of elements 3–350; the two test elements have finished by then. Two arrays of 60 run at once, within CERES's limit of about 250 across a user's arrays.

- An element in state `Eqw` failed to start: `qstat -j <job id>` says why (usually a path), and `qmod -c <job id>` clears it once fixed. `qdel <job id>` stops a whole array, `qdel <job id> -t 5-9` some of its elements.
- Runs of a stock-year need its exit settings, so a plan of runs is submitted once the exits plan has finished (or with `qsub -hold_jid <exits job id>`).
- `--count K` in `-v` (for example `-v CAMPAIGN=main,PLAN=runs,COUNT=4` with a quarter of the elements) makes each element do K tasks, for short tasks.
- An exits task is about 80 GP runs and a runs task 50, so each element takes minutes. A pooled ML-TFT run takes 30–60 minutes on the Mac, so plan it with `--block 1` (one run per task).
- The main campaign (the 50 main stocks, 50 runs per GP configuration) needs about 1,100 CPU hours at the Mac's speed in the pilot (several runs at once, after the speed-ups of 31 October), about 400 of them for A5 and A5-dalex; the hold-out stocks need as much again, and the robustness checks some hundreds more. With 60 elements at once the main stocks are roughly a day of waiting if CERES's older cores are as fast, two if they are half as fast. The test elements show the real speed.
- Results go to `~/rfgp-data/results/<campaign>/`. `collect` and the analysis can run on CERES or, after copying `results/<campaign>` back with `rsync`, on the Mac.
