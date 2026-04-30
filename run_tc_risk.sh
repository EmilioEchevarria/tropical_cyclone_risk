#PBS -P xv83
#PBS -N tc_risk
#PBS -q hugemem
#PBS -l ncpus=24
#PBS -l mem=460GB
#PBS -l jobfs=260GB
#PBS -l walltime=01:59:00
#PBS -l wd
#PBS -j oe
#PBS -l storage=gdata/v95+gdata/xv83

module load python3/3.11.0
module load eccodes
source /g/data/xv83/synthetic_TCs/tc_risk_venv/bin/activate

export HDF5_USE_FILE_LOCKING=FALSE

python run.py GL > log_tc_risk
