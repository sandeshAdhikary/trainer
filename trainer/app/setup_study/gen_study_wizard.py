import streamlit as st
from streamlit_ace import st_ace
import yaml
from copy import deepcopy

def check_config_wizard(check_config):
    if 'config' in st.session_state:
        st.subheader("Review Config")
        config = st_ace(yaml.dump(st.session_state.config),
                              language="yaml", 
                              theme="clouds_midnight"
                              )
        errs = check_config(config)
        

        hide_env_vars = st.checkbox("Use env variables to hide secrets", 
                                    key="hide_env_vars", value=True)
        
        final_config = deepcopy(yaml.safe_load(config))
        if hide_env_vars:
            env_vars = {}
            # Make the .env file
            if final_config['study_storage']['type'] == 'ssh':
                env_vars.update({
                    'SSH_HOST': final_config['study_storage']['host'],
                    'SSH_USERNAME': final_config['study_storage']['username'],
                    'SSH_PASSWORD': final_config['study_storage']['password'],
                    'SSH_DIR': final_config['study_storage']['root_dir']
                })
                # ${oc.env:SSH_HOST}
                final_config['study_storage']['host'] = '${oc.env:SSH_HOST}'
                final_config['study_storage']['username'] = '${oc.env:SSH_USERNAME}'
                final_config['study_storage']['password'] = '${oc.env:SSH_PASSWORD}'
                final_config['study_storage']['root_dir'] = '${oc.env:SSH_DIR}'
            
            if final_config['database']['type'] == 'mysql':
                env_vars.update({
                    'MYSQL_HOST': final_config['database']['host'],
                    'MYSQL_USERNAME': final_config['database']['username'],
                    'MYSQL_PASSWORD': final_config['database']['password'],
                    'MYSQL_DB': final_config['database']['name']
                })
                final_config['database']['host'] = '${oc.env:MYSQL_HOST}'
                final_config['database']['username'] = '${oc.env:MYSQL_USERNAME}'
                final_config['database']['password'] = '${oc.env:MYSQL_PASSWORD}'
                final_config['database']['name'] = '${oc.env:MYSQL_DB}'

            # Convert env_vars from dict to string 
            env_vars_str = ""
            for k, v in env_vars.items():
                env_vars_str += f"{k}={v}\n"

            st.download_button(
                "Download .env file",
                data=env_vars_str,
                file_name='.env',
                use_container_width=True,
                disabled=len(errs) > 0
            )
        st.download_button(
            "Download config file",
            data=yaml.dump(final_config),
            file_name='study_config.yaml',
            use_container_width=True,
            disabled=len(errs) > 0
        )
    else:
        st.error("Config Not Set!")

def gen_study_wizard():
    pass