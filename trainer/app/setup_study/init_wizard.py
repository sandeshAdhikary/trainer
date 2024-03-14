import streamlit as st
import yaml

def get_user_input():
    return st.selectbox("Create or Load Study?", options=['Create', 'Load'], index=0)

def make_config(user_input):
    config = None
    if user_input == 'Create':
        st.subheader("Create a New Study")
        study_name = st.text_input("What's the name of the study?", key='study_name', help="The name of the study",
                                    value=None)
        if study_name is not None:
            config = {
                'study_name': study_name,
                # 'study_storage': {},
                # 'database': {}
            }

    elif user_input == 'Load':
        st.subheader("Load an Existing Study")
        st.markdown("Ok, let's load an existing study. \
                    Do you want to enter the file path for the study's config file\
                    or upload the file directly?")
        config_file_input = st.radio("Config upload method", 
                                        options = ['Enter path', 'Upload file'], index=0,
                                        label_visibility='collapsed')
        
        if config_file_input == 'Enter path':
            config_file_path = st.text_input('Where is the config yaml file for the study?', 
                                                key='config_filepath', 
                                                help="The filepath of the config.yaml file",
                                                value=None
                                                )
            if config_file_path is not None:
                try:
                    config = yaml.load(open(config_file_path, 'r'), Loader=yaml.SafeLoader)
                except FileNotFoundError:
                    st.toast("File not found!", icon='❗')
        else:
            config_file = st.file_uploader("Upload the yaml config file for the study",
                                            accept_multiple_files=False,
                                            help='Upload the config.yaml file for the study.'
                                            )

            if config_file is not None:
                try:
                    config = yaml.load(config_file, Loader=yaml.SafeLoader)
                except FileNotFoundError:
                    st.toast("File not found!", icon='❗')

    else:
        ValueError(f"Unknown user input : {user_input}")

    return config

def submit_config(config):
    if config is None:
        config_errs = True
    else:
        st.markdown("This is what your study config looks like:")
        st.write(config)
        config_errs = config_errors(config)
        if not config_errs:
            st.toast("✅ Config passed initialization validation checks!")
            st.markdown("If everything looks good, click below to get started:")
        else:
            st.markdown("❗ Fix these config errors before proceeding!")
            st.error(config_errs)
    set_config_btn = st.button("Create Study Config", use_container_width=True, type='primary', disabled= config_errs!=False)
    return set_config_btn
        


def config_errors(config):
    config_errs = []
    if 'study_name' not in config:
        config_errs.append("Study name not set in config!")
    return False if len(config_errs) == 0 else config_errs

def init_wizard():
    st.subheader("Initialize Study")
    st.markdown("Let's start by defining the study.")
    with st.container(border=True):
        user_input = get_user_input()
    with st.container(border=True):
        config = make_config(user_input)
    with st.container(border=True):
        submit = submit_config(config)

    if submit:
        st.session_state['config'] = config
    return submit


        