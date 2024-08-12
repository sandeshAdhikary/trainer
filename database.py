import mysql.connector
from trainer.utils import generate_unique_hash
import pandas as pd
import json

class MySQLServer():
    def __init__(self, config):
        self.config = config      
        self.conn = mysql.connector.connect(
            host=config['host'],
            user=config['username'],
            password=config['password'],
        )
    def close(self):
        self.conn.close()

    def get_databases(self):
        """
        Get the list of databases on the server
        """
        cursor = self.conn.cursor()
        cursor.execute("SHOW DATABASES")
        db_list = cursor.fetchall()
        cursor.close()
        return db_list
    
    def check_studies_db(self):
        """
        Check if trainer_studies database exists
        """
        cursor = self.conn.cursor()
        cursor.execute("SHOW DATABASES LIKE 'trainer_studies'")
        res = cursor.fetchall()
        cursor.close()
        if len(res) < 1:
            return False
        return True
    
    def create_studies_db(self):
        """
        Create trainer_studies database (if it doesn't exist already)
        """
        if not self.check_studies_db():
            cursor = self.conn.cursor()
            cursor.execute("CREATE DATABASE trainer_studies")
            self.conn.commit()
            cursor.execute(f"GRANT ALL PRIVILEGES ON trainer_studies.* TO '{self.config['username']}'@'%' IDENTIFIED BY '{self.config['password']}';")
            cursor.execute("FLUSH PRIVILEGES")
            self.conn.commit()
            cursor.close()

        # Connect too the trainer_studies database
        db =  mysql.connector.connect(
                    host=self.config['host'],
                    user=self.config['username'],
                    password=self.config['password'],
                    database='trainer_studies'
                )
        
        cursor = db.cursor()
        cursor.execute("SHOW TABLES LIKE 'STUDIES'")
        studies_table = cursor.fetchall()
        if len(studies_table) < 1:
            # Create STUDIES table since it doesn't exist
            cursor.execute("""
                           CREATE TABLE STUDIES (
                               study_id VARCHAR(10) PRIMARY KEY,
                               study_name VARCHAR(100) NOT NULL,
                               config JSON NOT NULL
                           )
                           """)
            # Add example study
            self.add_example_study()
        cursor.close()
        db.close()


    def add_example_study(self):
        db =  mysql.connector.connect(
                    host=self.config['host'],
                    user=self.config['username'],
                    password=self.config['password'],
                    database='trainer_studies'
                )
        cursor = db.cursor()
        
        study_name = 'example_study'
        study_id = generate_unique_hash(study_name, include_time=False)
        config = json.dumps({
            'study_name': study_name,
            'study_id': study_id,
            'study_storage': {
                'root_dir': f'/tmp/{study_name}',
                'type': 'local'
            }
        })
        config = str(config)
        cursor.execute("""INSERT INTO STUDIES (study_id, study_name, config) 
                        VALUES (%s, %s, %s)""", (study_id, study_name, config))

        db.commit()
        cursor.close()
        db.close()

    def get_studies(self, format='dataframe'):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trainer_studies.STUDIES")
        studies = cursor.fetchall()
        cursor.close()
        if format=='dataframe':
            studies = pd.DataFrame(studies, 
                                   columns=['study_id', 'study_name',
                                            'config'])

        return studies

    def edit_study(self, study_info):
        """
        Edit existing study item
        """
        assert 'study_id' in study_info, "study_id not found in study_info"
        assert 'study_name' in study_info, "study_name not found in study_info"
        assert 'config' in study_info, "config not found in study_info"

        db =  mysql.connector.connect(
                    host=self.config['host'],
                    user=self.config['username'],
                    password=self.config['password'],
                    database='trainer_studies'
                )
        cursor = db.cursor()
        # cursor.execute("USE trainer_studies")
        # print(study_info['config'])
        # cursor.execute("SELECT * FROM trainer_studies.STUDIES WHERE study_id = %s", (study_info['study_id'],))
        out = cursor.execute("""UPDATE trainer_studies.STUDIES 
                          SET study_name = %s, config = %s
                          WHERE study_id = %s""", 
                          (study_info['study_name'], 
                           study_info['config'],
                           study_info['study_id']))
        # self.conn.commit()
        # out = cursor.fetchall()
        db.commit()
        cursor.close()
        db.close()
        return out

class DataBase():

    DB_TYPES = ['mysql']
    NO_SWEEP_ID = '89369c15ed' # Generated from generate_unique_hash('None')
    
    def __init__(self, config):
        assert config['type'] in self.DB_TYPES, f"Invalid DB type {config['type']}"

        # First test connection to MySQL server
        self._connect_to_server(config)

        # Reconnect -- this time to the database
        self.conn = mysql.connector.connect(
                    host=config['host'],
                    user=config['username'],
                    password=config['password'],
                    database=config['database']
                )
        
        assert self.conn.is_connected(), "Could not connect to database"

    def _connect_to_server(self, config):
        try:
            return MySQLServer({
                'host': config['host'],
                'username': config['username'],
                'password': config['password']
            })
        except Exception as e:
            return None
        

    def get_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("SHOW TABLES")
        return cursor.fetchall()
    
    def add_project(self, project_dict):
        project_name = project_dict['project_name']
        # If project ID not given, generate a new unique ID
        project_id = project_dict.get('project_id', 
                                      generate_unique_hash(project_name))

        cursor = self.conn.cursor()
        cursor.execute(f"""
                       INSERT INTO PROJECTS (project_name, project_id) 
                       VALUES ('{project_name}', '{project_id}')
                       ON DUPLICATE KEY UPDATE
                          project_name = '{project_name}'
                       """)
        self.conn.commit()
        cursor.close()

    def add_sweep(self, sweep_dict):
        sweep_name = sweep_dict['sweep_name']
        project_id = sweep_dict['project_id']
        # If sweep ID not given, generate a new unique ID
        sweep_id = sweep_dict.get('sweep_id', 
                                  generate_unique_hash(sweep_name))
        # If sweep_group_id not given, use sweep_id as sweep_group_id
        sweep_group_id = sweep_dict.get('sweep_group_id', None)
        if sweep_group_id is None:
            sweep_group_id = sweep_id
        cursor = self.conn.cursor()
        cursor.execute(f"""
                       INSERT INTO SWEEPS (sweep_name, sweep_id, sweep_group_id, project_id) 
                       VALUES ('{sweep_name}', '{sweep_id}', '{sweep_group_id}', '{project_id}')
                       ON DUPLICATE KEY UPDATE
                          sweep_name = '{sweep_name}',
                          sweep_group_id = '{sweep_group_id}',
                          project_id = '{project_id}'
                       """)
        self.conn.commit()
        cursor.close()

    def add_run(self, run_dict):
        run_id = run_dict['run_id']
        sweep_id = run_dict.get('sweep_id', self.NO_SWEEP_ID)
        project_id = run_dict['project_id']
        cursor = self.conn.cursor()
        cursor.execute(f"""
                       INSERT IGNORE INTO RUNS (run_id, sweep_id, project_id)
                       VALUES ('{run_id}', '{sweep_id}', '{project_id}')
                       """)
        self.conn.commit()
        cursor.close()
    
    def get_projects(self, format='dataframe'):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PROJECTS")
        output = cursor.fetchall()
        cursor.close()
        if format=='dataframe':
            output = pd.DataFrame(output, columns=['project_id', 'project_name'])
        return output

    def get_sweeps(self, project_id, format='dataframe'):
        cursor = self.conn.cursor()
        if project_id is not None:
            cursor.execute(f"SELECT * FROM SWEEPS WHERE project_id='{project_id}'")
        else:
            cursor.execute(f"SELECT * FROM SWEEPS")
        output = cursor.fetchall()
        cursor.close()
        if format=='dataframe':
            output = pd.DataFrame(output, 
                                  columns=['sweep_id', 'sweep_group_id',
                                            'project_id', 'sweep_name'])
        return output
    
    def close(self):
        self.conn.close()