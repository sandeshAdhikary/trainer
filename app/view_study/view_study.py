import streamlit as st
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv
import os
from trainer.app.defaults import ENV_FILE_PATH, STUDY_CONFIG_PATH
from trainer.database import DataBase
from trainer.storage import Storage
import pandas as pd
from trainer.utils import pretty_title


def project_page(project_id):
    # Get project info from database
    st.write(st.session_state['study_config'])
    db = DataBase(study_config['database'])
    # st.write(str(project_id[0]))
    st.write(project_id[0] == '6ff4535bdd')
    # sweeps = db.get_sweeps('6ff4535bdd')
    sweeps = db.get_sweeps(project_id[0])
    st.write(sweeps)
    db.close()

st.header("All Projects")

current_project_id = st.query_params.get_all('project_id')

with st.sidebar:
    config_path = st.text_input("Config Path", value=STUDY_CONFIG_PATH, help="Path to the study configuration file")
    env_path = st.text_input("Environment Path", value=ENV_FILE_PATH, help="Path to the environment file")

# Load env variables
load_dotenv(env_path)

# Load config file
study_config = OmegaConf.load(config_path)
study_config = DictConfig(OmegaConf.to_container(study_config, resolve=True))
st.session_state['study_config'] = study_config

# Connect to database
db = DataBase(study_config['database'])

# Show projects
projects = db.get_projects()
db.close()
# Check for icons
storage = Storage(study_config['study_storage'])

num_projects = len(projects)
num_rows = min(4, num_projects)
num_cols = (num_projects + num_rows - 1) // num_rows

idp = 0

if len(current_project_id) == 0:
    for idr in range(num_rows):
        # with st.container(border=True):
        cols = st.columns(num_cols)
        for col in cols:
            if idp <= num_projects - 1:
                project_name = projects['project_name'][idp]
                project_id = projects['project_id'][idp]
                with st.container(border=True):
                    st.markdown(f'<a href="/?project_id={project_id}" target="_self">{pretty_title(project_name)}</a>',unsafe_allow_html=True)
                idp += 1
else:
    project_page(current_project_id)
