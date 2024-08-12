import streamlit as st
import mysql.connector
import os
from trainer.utils import pretty_title
from streamlit_ace import st_ace
import json
import yaml
from omegaconf import OmegaConf, DictConfig
from trainer.database import MySQLServer

DB_TYPES = {'mysql': 0}

def get_user_input():
    default_choice = DB_TYPES.get(st.session_state['config']['database'].get('type', 'local'), 'mysql')
    return st.selectbox("Pick your database type:", 
                        options=DB_TYPES.keys(), 
                        index=default_choice,
                        help='Pick "mysql" to store on a MySQL database')


def make_config(user_input):
    config = {}
    if user_input == 'mysql':
        st.subheader("MYSQL Database Connection")
        st.markdown("Ok, let's set up the MYSQL deets. Here's what the current config looks like.\
                    Make changes and hit Enter to apply. Once you're done, press 'Update Study Config' to commit changes.")
        config['type'] = 'mysql'
        
        config['name'] = st.text_input("Database Name",
                                       value=st.session_state['config']['study_name'],
                                       disabled=False,
                                       help="The name of the database to store the study data.\
                                        Generally, would be same as study_name"
                                       )
        config['host'] = st.text_input("Database Host",
                                       value=st.session_state['config']['database'].get('host', None),
                                       help="The IP address or domain name of the remote server",
                                       type='password')
        config['username'] = st.text_input("SSH Username",
                                           value=st.session_state['config']['database'].get('username', None),
                                           help="The username to use to connect to the database",
                                           type='password')
        config['password'] = st.text_input("SSH Password",
                                           value=st.session_state['config']['database'].get('password', None),
                                           help="The password to use to connect to the database",
                                           type='password')

    else:
        raise NotImplementedError("Only MySQL is supported at the moment")
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
    config_errs= []
    return False if len(config_errs) == 0 else config_errs


def test_connection(config, create_new_database=False):
    conn = None
    if config['type'] == 'mysql':
        host = config['host']
        username = config['username']
        password = config['password']
        try:
            conn = mysql.connector.connect(
                host=host,
                user=username,
                password=password,
            )
            cur = conn.cursor()
            st.toast("Successfully connected to host", icon='✅')
        except: 
            st.toast('Could not connect. Check host/username/password !', icon='❗')
            return False
        
        try:
            query = f"SHOW DATABASES LIKE '{config['name']}'"
            cur.execute(query)
            res = [x[0] for x in cur.fetchall()]
            if config['name'] in res:
                st.toast("Database found!", icon='✅')
            else:
                st.error(f"Couldn't find database {config['name']}. \
                         If it doesn't exist, create the database (and grant access to {username}) and try again.")
                st.toast("Database not found!", icon='❗')
                return False
        except Exception as e:
            st.error(e)
            return False

    else:
        raise NotImplementedError("Only MySQL is supported at the moment")
    
    if conn is not None:
        conn.close()
    
    return True

def database_wizard():

    if 'config' not in st.session_state:
        st.session_state['config'] = {}

    if 'database' not in st.session_state['config']:
        st.session_state['config']['database'] = {}
        
    if 'mysql_server' not in st.session_state:
        st.session_state['mysql_server'] = None

    st.subheader("Setup Database")
    st.markdown("""
                First off, let's connect to a MYSQL server.
                """)
    with st.popover("Set up MYSQL Connection"):
        with st.form(key='mysql_creds'):
            host = st.text_input("MYSQL Host", value=os.environ.get('MYSQL_HOST'), type='password')
            username = st.text_input("MYSQL Username", value=os.environ.get('MYSQL_USERNAME'), type='password')
            pwd = st.text_input("MYSQL password", value=os.environ.get('MYSQL_PASSWORD'), type='password')

            submit = st.form_submit_button("Connect to MYSQL Server")
            if submit:
                try:
                    st.session_state['mysql_server'] = MySQLServer(
                        {'host': host,
                        'username': username,
                        'password': pwd
                        })
                    st.success("✅ Connected to MySQL server")
                except Exception as e:
                    st.error(f"Could not connect to MySQL server. ERROR {e}")
                    st.session_state['mysql_server'] = None

    if st.session_state['mysql_server'] is not None:
        st.success("✅ MySQL server connected")
        mysql_server = st.session_state['mysql_server']
        studies = mysql_server.get_studies()

        for study_id in studies['study_id'].unique():
            study = studies[studies['study_id'] == study_id]
            config = json.loads(study['config'][0])
            study_name = study['study_name'][0]
            title = f'{pretty_title(study_name)}'
            with st.expander(title):
                st.header(title)
                st.markdown(f"Study ID: {study_id}")
                config = st_ace(yaml.dump(config),
                                    language="yaml", 
                                    theme="clouds_midnight"
                                    )
                config = yaml.load(config, Loader=yaml.SafeLoader)
                with st.form(key=f'form_{study_id}'):
                    submit = st.form_submit_button("Update Study Config")
                    if submit:
                        try:
                            # Update the study config
                            st.session_state['mysql_server'].edit_study({
                                'study_id': str(study_id),
                                'study_name': str(study_name),
                                'config': json.dumps(config)
                            })
                            st.toast("Study config updated!", icon='✅')
                        except Exception as e:
                            st.error(f"Could not update study config. ERROR {e}")
    else:
        st.error("MySQL server not established! Set up credentials.")

    

    # Show the studies table for the database
    

    # Create studies db if it doesn't exist
    # db_server.create_studies_db()
    
    
    # Connect to a MYSQL server

    # server has a database called 'trainer' which lists all studies

    # Each study is its own database
    
    # with st.container(border=True):
    #     user_input = get_user_input()
    # with st.container(border=True):
    #     config = make_config(user_input)
    # with st.container(border=True):
    #     submit = submit_config(config)

    #     test_connection_check = st.checkbox("Test connection before update", key="test_connection", value=True)

    #     if submit:
    #         connection_valid = True
    #         if test_connection_check:
    #             connection_valid = test_connection(config)
    #         if connection_valid:
    #             st.session_state['config']['database'].update(config)
    #             st.toast("Config Updated!", icon='✅')
    #         else:
    #             st.toast("Config not updated.", icon='❗')
    # return submit

