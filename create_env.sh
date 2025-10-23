module load python-scientific/3.11.5-foss-2023b
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt