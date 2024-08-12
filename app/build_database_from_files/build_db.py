from trainer.storage import Storage
import dotenv
from trainer.app.defaults import ENV_FILE_PATH, STUDY_CONFIG_PATH
import os
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import hashlib
import time
from trainer.database import DataBase
from trainer.utils import generate_unique_hash
dotenv.load_dotenv(ENV_FILE_PATH)
study_config = OmegaConf.load(STUDY_CONFIG_PATH)

def add_from_storage_to_db():
    """
    Loop over folders in the study and populate them into the database
    We'll assign unique IDs (run_id, sweep_id, project_id) based on the current time
    """
    # Open connection to study storage
    storage_config = study_config['study_storage']
    storage_config = OmegaConf.to_container(storage_config, resolve=True)
    storage = Storage(storage_config)

    runs_db = []
    for project in storage.get_filenames():
        sweeps_and_runs = storage.get_filenames(project)
        for sr in sweeps_and_runs:
            sr_split = sr.split('sweep_')
            if len(sr_split) > 1:
                sweep_name = sr_split[1]
                # Get runs in the sweep
                runs = storage.get_filenames(f"{project}/{sr}")
                for run in runs:
                    runs_db.append({'run_id': run, 'sweep_name': sweep_name, 'project_name': project})
            else:
                runs_db.append({'run_id': sr, 'sweep_name': None, 'project_name': project})
    runs_db = pd.DataFrame(runs_db)

    # Get unique IDs
    sweep_ids = {x:generate_unique_hash(x) for x in runs_db['sweep_name'].unique()}
    runs_db['sweep_id'] = runs_db['sweep_name'].apply(lambda x: sweep_ids[x] if x else None)
    project_ids = {x:generate_unique_hash(x) for x in runs_db['project_name'].unique()}
    runs_db['project_id'] = runs_db['project_name'].apply(lambda x: project_ids[x] if x else None)

    # Connect to database
    # Open connection to study storage
    db_config = study_config['database']
    db_config = OmegaConf.to_container(db_config, resolve=True)
    db = DataBase(db_config)

    projects = runs_db[['project_name', 'project_id']].drop_duplicates().reset_index(drop=True)

    for proj in projects.iterrows():
        db.add_project({
            'project_name': proj[1]['project_name'],
            'project_id': proj[1]['project_id']
            })
        
    # # Add sweeps
    sweeps = runs_db[['sweep_name', 'sweep_id', 'project_id']].drop_duplicates().reset_index(drop=True)
    for sweep in sweeps.iterrows():
        sweep_name = sweep[1]['sweep_name']
        if sweep_name is not None:
            db.add_sweep({
                'sweep_name': sweep[1]['sweep_name'],
                'sweep_id': sweep[1]['sweep_id'],
                'project_id': sweep[1]['project_id']
            })

    # # Add runs
    runs = runs_db[['run_id', 'sweep_name', 'sweep_id', 'project_name', 'project_id']].reset_index(drop=True)
    for run in runs.iterrows():
        run_id = run[1]['run_id']
        db.add_run({
            'run_id': run[1]['run_id'],
            'sweep_id': run[1]['sweep_id'],
            'project_id': run[1]['project_id']
        })

    # generate_unique_hash('NoSweepID')

    # print("Done")
        
add_from_storage_to_db()