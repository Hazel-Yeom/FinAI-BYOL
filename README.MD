#### 1. Create venv and Install requirements.txt


#### 2. Download NASDAQ data
    python fetch_universe_and_prices.py


#### 3. Start model training
    python byol_tuning.py --n_trials 20 --epochs 10


#### 4. Check your model performance
    check_model_performance.ipynb





python byol_v03_tuning.py --study_name byol_v03_tuning_XX --n_trials 20 --epochs 5 --train_final --final_epochs 20
    python byol_v03_tuning.py `
    --study_name byol_v03_tuning_XX `
    --n_trials 20 `
    --epochs 5 `
    --timeout_hours 100000 `
    --train_final `
    --final_epochs 10 `