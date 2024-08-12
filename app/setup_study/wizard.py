import streamlit as st
import extra_streamlit_components as stx
from streamlit_ace import st_ace
import json
import yaml
import time
from trainer.storage import Storage
from streamlit_custom_notification_box import custom_notification_box as st_notification
from streamlit_extras.stylable_container import stylable_container 
from trainer.app.setup_study.study_storage_wizard import study_storage_wizard
from trainer.app.setup_study.database_wizard import database_wizard
from trainer.app.setup_study.init_wizard import init_wizard
from trainer.app.setup_study.gen_study_wizard import check_config_wizard, gen_study_wizard
from trainer.app.setup_study.sql_server_wizard import sql_server_wizard
from trainer.app.setup_study.setup_model_wizard import setup_model_wizard
from functools import partial
import os
from trainer.utils import load_env
load_env() # Load env variables from .env file

def check_config(config=None):
    config = config or st.session_state.get('config', {})
    errs = []
    # Study storage check
    if 'study_storage' not in config:
        errs.append("Study Storage not set in config!")
    
    if 'database' not in config:
        errs.append("Database not set in current config!")
        
    for err in errs:
        st.error(err)

    return errs

if __name__ == '__main__':

    if 'config' not in st.session_state:
        st.session_state['config'] = {}

    wizard_steps = {
        'database' : {
            'title': 'SQL Server Setup',
            'num': 0,
            'func': sql_server_wizard
        },
        'study_storage' : {
            'title': 'Study Storage',
            'num': 1,
            'func': study_storage_wizard
        },
        'setup_model': {
            'title': 'Setup Model',
            'num': 2,
            'func': setup_model_wizard
        }
        # 'study_structure': {
        #     'title': 'Study Structure',
        #     'num': 2,
        #     'func': study_structure_wizard
        # },
        # }
        # 'init' : {
        #         'title': 'Initialization',
        #         'num': 1,
        #         'func': init_wizard
        #     },
        # 'check_config': {
        #     'title': 'Check Config',
        #     'num': 3,
        #     'func': partial(check_config_wizard, check_config)
        # },
        # 'gen_study': {
        #     'title': 'Generate Study',
        #     'num': 4,
        #     'func': gen_study_wizard
        # }
    }


wizard_steps = sorted([step_val for step_name, step_val in wizard_steps.items()], key=lambda x: x['num'])
step_titles = [x['title'] for x in wizard_steps]

if 'current_step' not in st.session_state:
    st.session_state.current_step = 'home'

with st.sidebar:
    # Stepper
    step_id = stx.stepper_bar(step_titles, is_vertical=True, lock_sequence=False)
    check_config()

# Run view based on step
submit = wizard_steps[step_id]['func']()
