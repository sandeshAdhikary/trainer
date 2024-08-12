import streamlit as st
import extra_streamlit_components as stx
from streamlit_ace import st_ace
import json
import yaml
import time
from trainer.storage import Storage
from streamlit_custom_notification_box import custom_notification_box as st_notification
from streamlit_extras.stylable_container import stylable_container 

container_css_styles = {
    'main' : """
            {
                background-color: rgba(10, 10, 10, 0.2);
            }
            """
}

class Wizard():
    def __init__(self, wizard_config):
        self.steps = wizard_config['steps']
        self.steps.update({
            'init' : {
                'title': 'Initialization',
                'num': 0,
                'func': self.init_study
            },
            'wizard_config': {
                'title': 'Check Config',
                'num': len(self.steps),
                'func': self.final_config
            }
        })
        self.num_steps = len(self.steps)
        if 'current_view' not in st.session_state:
            st.session_state['current_view'] = 'Home'

        self.sidebar = st.sidebar
        

    def run(self):

        steps = sorted([step_val for step_name, step_val in self.steps.items()], key=lambda x: x['num'])
        step_titles = [x['title'] for x in steps]

        with self.sidebar:
            # Stepper
            step_id = stx.stepper_bar(step_titles, is_vertical=True, lock_sequence=False)
            self._check_config()
        # Run view based on step
        steps[step_id]['func']()



    def home(self):
        st.header("Setup Wizard")
        
    def _check_config(self):
        config = st.session_state.get('config', {})
        # Study storage check
        if 'study_storage' not in config:
            st.error("Study Storage not set in config!")
        
        if 'database' not in config:
            st.error("Database not set in current config!")

        if len(config.keys()) > 1:
            # Storage check
            assert 'study_storage' in config, "Study storage not set"


    def final_config(self):
        final_config = st_ace(value=json.dumps(st.session_state['config'], indent=2), 
                              language="yaml", 
                              theme="clouds_midnight"
                              )
        

    def init_study(self):
        st.subheader("Initialize Study")

        st.markdown("Let's start by defining the study")
        with st.container(border=True):
            init_study_input = st.selectbox("Create or Load Study?", options=['Create', 'Load'], index=0)
        
        with st.container(border=True):
            form_output = self._init_study_form_view(init_study_input)

    def _init_study_form_view(self, init_study_input):
        output = {'config': None}
        if init_study_input == 'Create':
            st.subheader("Create a New Study")
            study_name = st.text_input("What's the name of the study?", key='study_name', help="The name of the study",
                                       value=None)
            if study_name is not None:
                output['config'] = {
                    'study_name': study_name,
                    # 'study_storage': {},
                    # 'database': {}
                }
        elif init_study_input == 'Load':
            st.subheader("Load an Existing Study")
            st.markdown("Ok, let's load an existing study. \
                        Pick whether you want to enter the file path for the study's config file\
                        or upload the file directly.")
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
                        output['config'] = yaml.load(open(config_file_path, 'r'), Loader=yaml.SafeLoader)
                    except FileNotFoundError:
                        st.toast("File not found!", icon='❗')


            else:
                config_file = st.file_uploader("Upload the yaml config file for the study",
                                                accept_multiple_files=False,
                                                )

                if config_file is not None:
                    try:
                        output['config'] = yaml.load(config_file, Loader=yaml.SafeLoader)
                    except FileNotFoundError:
                        st.toast("File not found!", icon='❗')
                
        if output['config'] is not None:
            st.markdown("Here's what your study's config looks like")
            st.write(output['config'])

        with st.container(border=True):
            set_config_btn = st.button("Create Study Config", use_container_width=True, type='primary')

            if set_config_btn:
                st.session_state['config'] = output['config']
                st.toast("Config Set!", icon='✅')
            # test_connection_check = st.checkbox("Test connection before update", key="test_connection", value=True)

        return output
            




if __name__ == "__main__":

    class MyWiz(Wizard):
        def __init__(self):
            config = {
                'steps': {
                    'study_storage' : {
                        'title': 'Study Storage',
                        'num': 1,
                        'func': self.train_storage
                    },
                    'database' : {
                        'title': 'Database',
                        'num': 2,
                        'func': self.database
                    },
                    'logger' : {
                        'title': 'Logger',
                        'num': 3,
                        'func': self.logger
                    }
                }
            }
            super().__init__(config)
            st.session_state['setup_config'] = {}

        def train_storage(self):
            st.subheader("Setup Study Storage")

            storage_types = {'local': 0, 'ssh': 1}
            
            # with stylable_container(css_styles=container_css_styles['main'], key='train_storage_container'):
            with st.container(border=True):
                with st.container(border=True):           
                    study_storage = st.selectbox("Where do you want to store the study?", 
                                                 options=storage_types.keys(), 
                                                 index=storage_types[st.session_state['config']['study_storage'].get('type', 'local')])

                with st.container(border=True):
                    form_output = self._train_storage_form_view(study_storage)
                    
                with st.container(border=True):
                    update_config_btn = st.button("Update Study Config", use_container_width=True, type='primary')
                    test_connection_check = st.checkbox("Test connection before update", key="test_connection", value=True)

                    if update_config_btn:
                        connection_valid = True
                        if test_connection_check:
                            connection_valid = self._test_storage_connection(form_output['study_storage'])

                        if connection_valid:
                            self._update_config(form_output)
                            st.toast("Config Updated!", icon='✅')
                        else:
                            st.toast("Config not updated!", icon='❌')
            
        def _train_storage_form_view(self, study_storage):
            # output = {'study_storage': study_storage}
            output = {'study_storage': {}}
            if study_storage == "local":
                st.subheader("Setup Local Storage")
                st.markdown("Ok, let's set up the local storage. Here's what the current config looks like.\
                            Make changes and hit Enter to apply. Once you're done, press 'Update Study Config' to commit changes.")
                output['study_storage']['type'] = 'local'
                output['study_storage']['root_dir'] = st.text_input("Directory to store study data", 
                                                                    value=st.session_state['config']['study_storage'].get('root_dir', None),
                                                                    help="The directory to store the study data")

            if study_storage == 'ssh':
                st.subheader("Setup SSH Storage")
                st.markdown("Ok, let's set up the SSH deets. Here's what the current config looks like.\
                            Make changes and hit Enter to apply. Once you're done, press 'Update Study Config' to commit changes.")
                output['study_storage']['type'] = 'ssh'
                
                output['study_storage']['host'] = st.text_input("SSH Host", 
                                                                value=st.session_state['config']['study_storage'].get('host', None),
                                                                help="The IP address or domain name of the remote server",
                                                                type='password')
                output['study_storage']['username'] = st.text_input("SSH Username", 
                                                                    value=st.session_state['config']['study_storage'].get('username', None),
                                                                    help="The username to use to connect to the remote server",
                                                                    type='password')
                output['study_storage']['password'] = st.text_input("SSH Password", 
                                                                    value=st.session_state['config']['study_storage'].get('password', None),
                                                                    help="The password to use to connect to the remote server",
                                                                    type='password')
                output['study_storage']['root_dir'] = st.text_input("Directory to store study data", 
                                                                    value=st.session_state['config']['study_storage'].get('root_dir', None),
                                                                    help="The directory on the remote server to store the study data")
                output['study_storage']['overwrite'] = False
            
            output['study_storage'].update(output['study_storage'])
                        
            return output

        def _update_config(self, new_config):
            st.session_state['config'].update(new_config)


        def _test_storage_connection(self, storage_config):
            connection_valid = False
            try:
                storage = Storage(storage_config)
                storage.close_connection()
                connection_valid = True
                st.toast("Test connection successful!", icon='✅')
            except Exception as e:
                # st.write('Error')
                st.toast(f"Could not establish connection. Error: {e}", icon='❗')
            return connection_valid

        
        def eval_storage(self):
            st.write("Eval Storage")
        
        def database(self):
            st.subheader("Setup Database")

            db_types = {'mysql': 0, 'sqlite': 1, 'None': 2}
            with st.container(border=True):

                with st.container(border=True):
                    db_type = st.selectbox("What type of database do you want to use?", 
                                           options=db_types.keys(), 
                                           index=db_types[st.session_state['config']['database'].get('type', 'None')])
                    
                with st.container(border=True):
                    self._database_form_view(db_type)

        def _database_form_view(self, db_type):
            output = {'database': {}}
            if db_type == 'None':
                st.write("No database selected. \
                            You can skip this for now but\
                            you'll need to configure a database if you want to evaluate models later.")
                output['database'] = None
            elif db_type == 'sqlite':
                st.write("SQLite databases are not supported right now. Use MySQL database.")
            elif db_type == 'mysql':
                st.markdown("Ok. Let's set up the MySQL deets.")
                st.markdown("Here's what the current config looks like.\
                            Make changes and hit Enter to apply. Once you're done, press 'Update Study Config' to commit changes.")
                output['database']['type'] = 'mysql'
                output['database']['name'] = st.text_input("Database Name", 
                                                            value=st.session_state['config']['database'].get('name', None),
                                                            help="The name of the database to use")
                output['database']['host'] = st.text_input("Database Host",
                                                            value=st.session_state['config']['database'].get('host', None),
                                                            help="The IP address or domain name of the database server",
                                                            type='password')
                output['database']['username'] = st.text_input("Database Username",
                                                                value=st.session_state['config']['database'].get('username', None),
                                                                help="The username to use to connect to the database",
                                                                type='password')
                output['database']['password'] = st.text_input("Database Password",
                                                                value=st.session_state['config']['database'].get('password', None),
                                                                help="The password to use to connect to the database",
                                                                type='password')
                


        def logger(self):
            st.write("Logger")

    wiz = MyWiz()
    

    wiz.run()