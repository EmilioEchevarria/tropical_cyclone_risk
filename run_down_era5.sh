#PBS -P xv83
#PBS -N down_era5
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=192GB
#PBS -l walltime=09:20:00
#PBS -l wd
#PBS -j oe
#PBS -l storage=gdata/v95+gdata/xv83

module load python3/3.11.0
module load eccodes
source /g/data/xv83/synthetic_TCs/tc_risk_venv/bin/activate

python scripts/download_era5_netcdf.py
