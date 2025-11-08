# tdt17-task4-football-2025
TDT17 project for football player/referee/ball detection on the IDUN dataset.

# IDUN setup for `tdt17-task4-football-2025`

## 2. Go to IDUN and prepare the folder

On IDUN:

```bash
ssh eliasls@idun.hpc.ntnu.no
cd /cluster/work/eliasls/tdt17/task4
```
Clone your repo into that folder (after you’ve added an SSH key for IDUN to GitHub):

git clone git@github.com:eliasls/tdt17-task4-football-2025.git .

If SSH to GitHub doesn’t work yet, use HTTPS or add an SSH key on IDUN and paste it into GitHub.

Now /cluster/work/eliasls/tdt17/task4 = your project.

3. Create the conda environment (IDUN way)
In your project folder on IDUN:

bash
Kopier kode
module load Anaconda3/2023.09-0   # or whatever version exists
conda create -n tdt17 python=3.11
conda activate tdt17
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name=tdt17
Install basic packages:

bash
Kopier kode
pip install opencv-python matplotlib pandas
(Optional) on your laptop create a requirements.txt:

text
Kopier kode
opencv-python
matplotlib
pandas
and commit/push it.

4. Test Jupyter on IDUN
From the same project folder on IDUN:

bash
Kopier kode
jupyter notebook --no-browser
Then do the SSH tunnels exactly like in the NTNU tutorial and open the URL in your browser.
Pick the kernel tdt17.

If that works → ✅ you have environment + notebook running on the cluster.

5. Create the first notebook: notebooks/01_explore_dataset.ipynb
Goal: prove you can read from the football dataset on IDUN.

Inside the notebook:

python
Kopier kode
import os

DATA_ROOT = "/cluster/projects/vc/courses/TDT17/other/Football2025"  # check the real path
files = os.listdir(DATA_ROOT)
files[:20]
Then:

find one image

load it with cv2.imread(...)

show it with matplotlib

If there are annotation files, open one and print it. This is your “dataset understanding” part — keep this notebook in the repo.

6. Divide roles early
Example:

Person A: notebooks, EDA, visualization, sanity-checking labels

Person B: src/train.py, slurm/train.slurm, configs/

So you don’t both edit the same notebook at the same time.

7. Create a minimal Slurm file
In slurm/train.slurm:

bash
Kopier kode
#!/bin/bash
#SBATCH --job-name=tdt17-test
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --partition=CPUQ
#SBATCH --output=slurm_out.txt

module load Anaconda3/2023.09-0
conda activate tdt17
python src/hello.py
In src/hello.py:

python
Kopier kode
print("IDUN works!")
Run:

bash
Kopier kode
sbatch slurm/train.slurm
If it runs → you now have the whole pipeline: GitHub → IDUN → Slurm.

8. Push/pull routine (very important when you’re two)
Always git pull on IDUN before running a notebook or a Slurm job → so you run the latest code.

Always git pull on your laptop before editing a notebook → so you don’t overwrite your teammate.

If you both must edit the same notebook: make a copy like 01_explore_dataset_elias.ipynb and merge later.

Put this in the main README.md so you both remember.

9. Document IDUN specifics in the README
Add something like:

markdown
Kopier kode
## Cluster setup (IDUN)
module load Anaconda3/2023.09-0
conda activate tdt17

## Start Jupyter on IDUN
jupyter notebook --no-browser
# do SSH tunnel as in NTNU IDUN tutorial
So the TA (and your teammate) knows how to run it.

10. First “real” code
After the EDA notebook works:

src/dataset_utils.py – functions to read images + labels

configs/football.yaml – dataset path, classes, image size

later: src/train.py – actual training / evaluation code

later: Slurm script for training on GPU