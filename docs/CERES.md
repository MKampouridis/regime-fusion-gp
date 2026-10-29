# Running on CERES

CERES (University of Essex) is a Rocks 7 cluster (CentOS 7) with **Sun Grid Engine**: jobs are submitted with `qsub`, and job arrays number their elements with `SGE_TASK_ID`, from 1. The home folder (`/home/<user>`, on shared storage) is visible from every node. The system Python is 3.6, too old for this code (3.11 or later), so the study brings its own (step 3).

A campaign is a list of numbered tasks ([`ENGINE.md`](ENGINE.md), Campaigns); each element of a job array does one of them. Tasks can be repeated safely, so a failed or stopped element is just submitted again.

## 1. What CERES needs to tell us first

Run on the login node and send the output:

```bash
ldd --version | head -1; cat /etc/redhat-release
module show opt-python 2>&1 | head -20; (module load opt-python 2>/dev/null && python3 --version)
which apptainer singularity git curl 2>&1 | grep -v "no "
curl -sS -o /dev/null -w "conda-forge %{http_code}\n" https://conda.anaconda.org/conda-forge/
curl -sS -o /dev/null -w "github %{http_code}\n" https://github.com
qconf -sql; qconf -spl; qstat -g c
qconf -sq all.q 2>/dev/null | grep -E "^(qname|slots|h_rt|s_rt|h_vmem|pe_list)"
qconf -sconf | grep -E "max_aj_instances|max_aj_tasks"
qhost | head -5
```

They show the C library (which decides which Python packages can run), whether a newer Python or a container system is offered, whether the login node can download packages, and the queues' limits (run time, memory, the largest job array).

## 2. The code

The repository is private, so CERES needs read access. The tidiest way is a read-only deploy key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/rfgp_deploy -N ""        # on CERES
cat ~/.ssh/rfgp_deploy.pub                               # add on GitHub: repository Settings > Deploy keys (read only)
printf 'Host github.com\n  IdentityFile ~/.ssh/rfgp_deploy\n' >> ~/.ssh/config
git clone git@github.com:MKampouridis/regime-fusion-gp.git ~/regime-fusion-gp
```

`git pull` then brings later changes. (Without GitHub access: `git bundle create rfgp.bundle --all` on the Mac, copy the file over, and `git clone rfgp.bundle ~/regime-fusion-gp`.)

## 3. The environment

```bash
cd ~/regime-fusion-gp && bash scripts/ceres/setup.sh
```

It installs Miniforge in `~/miniforge3`, creates the environment `rfgp` from conda-forge (Python 3.12 and the study's packages, PyTorch if it installs), installs the package and runs the tests. conda installs only builds that work with the cluster's C library, so versions may be older than on the Mac; the tests show whether they behave the same.

## 4. The data

Runs read the prices, the processed indicators and the regimes, about 80 MB. From the Mac (`<ceres>` being the login address):

```bash
rsync -av ~/rfgp-data/interim/prices ~/rfgp-data/interim/news <ceres>:rfgp-data/interim/
rsync -av ~/rfgp-data/processed <ceres>:rfgp-data/
```

Licensed data (FNSPID) stay within the university's systems, which its licence allows for research.

## 5. A campaign

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rfgp
export RFGP_DATA_DIR=~/rfgp-data
cd ~/regime-fusion-gp && mkdir -p logs
python -m regime_fusion.engine tasks --campaign main --kind exits     # prints the qsub line to use
qsub -t 1-350 -tc 100 -v CAMPAIGN=main,PLAN=exits scripts/ceres/tasks.sge
qstat                                                                  # what is running
python -m regime_fusion.engine status --campaign main                  # what is finished
```

Each element may run up to 4 hours (`h_rt` in `scripts/ceres/tasks.sge`; change it to CERES's limits once known): an exits task is about 80 GP runs, a runs task 30, both well under it. A pooled ML-TFT run takes 30–60 minutes, so plan it with `--block 1` (one run per task).

Runs of a stock-year need its exit settings, so a plan of runs is submitted once the exits plan has finished (or with `qsub -hold_jid <exits job id>`). `--count K` in `-v` (for example `-v CAMPAIGN=main,PLAN=runs,COUNT=4` with a quarter of the elements) makes each element do K tasks, for short tasks or arrays near the cluster's limit. Results go to `~/rfgp-data/results/<campaign>/`; `collect` and the analysis can run on CERES or, after copying `results/<campaign>` back with `rsync`, on the Mac.
