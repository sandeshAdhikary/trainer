import streamlit as st
from trainer.database import MySQLServer
import streamlit as st
import os
from trainer.utils import pretty_title
from streamlit_ace import st_ace
import json
import yaml

def get_user_input():
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

def sql_server_wizard():
    if 'mysql_server' not in st.session_state:
        st.session_state['mysql_server'] = None
    if 'studies' not in st.session_state:
        st.session_state['studies'] = None

    # User input
    get_user_input()

    st.markdown("Here are the studies in your database")
    if st.session_state['mysql_server'] is not None:
        st.toast("MySQL server connected", icon='✅')
        mysql_server = st.session_state['mysql_server']
        studies = mysql_server.get_studies()
        st.session_state['studies'] = {study_id: study_name for study_id, study_name in zip(studies['study_id'], studies['study_name'])}
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

