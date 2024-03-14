import streamlit as st
from trainer.storage import Storage
from trainer.utils import pretty_title

STORAGE_TYPES = {
      'local': 0,
      'ssh': 1,
}

def get_user_input():
    return st.selectbox("Where do you want to store the study?", 
                        options=STORAGE_TYPES.keys(), 
                        index=STORAGE_TYPES[st.session_state['config']['study_storage'].get('type', 'local')],
                        help='Pick "local" to store on current machine or "ssh" to store on a remote machine')

def make_config(user_input):
    config = {}
    if user_input == 'local':
        st.subheader("Setup Local Storage")
        st.markdown("Ok, let's set up the local storage. Here's what the current config looks like.\
                    Make changes and hit Enter to apply. Once you're done, press 'Update Study Config' to commit changes.")
        config['type'] = 'local'
        config['root_dir'] = st.text_input("Directory to store study data", 
                                           value=st.session_state['config']['study_storage'].get('root_dir', None),
                                            help="The directory to store the study data")
    elif user_input == 'ssh':
        st.subheader("Setup SSH Storage")
        st.markdown("Ok, let's set up the SSH deets. Here's what the current config looks like.\
                    Make changes and hit Enter to apply. Once you're done, press 'Update Study Config' to commit changes.")
        config['type'] = 'ssh'
        
        config['host'] = st.text_input("SSH Host", 
                                       value=st.session_state['config']['study_storage'].get('host', None),
                                       help="The IP address or domain name of the remote server",
                                       type='password')
        config['username'] = st.text_input("SSH Username", 
                                           value=st.session_state['config']['study_storage'].get('username', None),
                                           help="The username to use to connect to the remote server",
                                           type='password')
        config['password'] = st.text_input("SSH Password", 
                                           value=st.session_state['config']['study_storage'].get('password', None),
                                           help="The password to use to connect to the remote server",
                                           type='password')
        config['root_dir'] = st.text_input("Directory to store study data", 
                                           value=st.session_state['config']['study_storage'].get('root_dir', None),
                                           help="The directory on the remote server to store the study data")
        config['overwrite'] = False
    
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
    set_config_btn = st.button("Update Study Config", use_container_width=True, type='primary', disabled= config_errs!=False)
    return set_config_btn


def config_errors(config):
    config_errs = []
    return False if len(config_errs) == 0 else config_errs

def test_storage_connection(config):
    connection_valid = False
    try:
        storage = Storage(config)
        storage.close_connection()
        connection_valid = True
        st.toast("Test connection successful!", icon='✅')
    except Exception as e:
        # st.write('Error')
        st.toast(f"Could not establish connection. Error: {e}", icon='❗')
    return connection_valid

def study_storage_wizard():
    submit = False
    if 'studies' not in st.session_state:
        st.error("Set up SQL Server to get a list of studies first!")
    else:
        if 'study_storage' not in st.session_state['config']:
            st.session_state['config']['study_storage'] = {}

        st.subheader("Setup Study Storage")

        with st.popover("Select Study"):
            study_ids = list(st.session_state['studies'].keys())
            # st.write(st.session_state['studies'])
            study_id = st.selectbox("Select Study", options=study_ids, help="Select the study to update",
                         format_func = lambda x: st.session_state['studies'][x])
            study_name = st.session_state['studies'][study_id]

        st.subheader(f"{pretty_title(study_name)}")

        with st.container(border=True):
            user_input = get_user_input()
        with st.container(border=True):
            config = make_config(user_input)
        with st.container(border=True):
            submit = submit_config(config)

            test_connection_check = st.checkbox("Test connection before update", 
                                                key="test_connection",
                                                value=True
                                                )
            if submit:
                connection_valid = True
                if test_connection_check:
                    connection_valid = test_storage_connection(config)
                if connection_valid:
                    st.session_state['config']['study_storage'].update(config)
                    st.toast("Config Updated!", icon='✅')
                else:
                    st.toast("Config not updated.", icon='❗')

            
    return submit