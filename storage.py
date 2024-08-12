from abc import ABC
import tempfile
import os
import wandb
import zipfile
import paramiko
import pickle
import torch
import numpy as np
import shutil
import zipfile
import yaml
from envyaml import EnvYAML
import glob
import io
import mysql.connector
import os
import json
import hashlib
from io import BytesIO
import numpy as np
import pandas as pd
import logging
from copy import copy, deepcopy
import pandas as pd
from datetime import datetime, timezone, timedelta
from time import sleep
from rich.progress import Progress
from stat import S_ISDIR
import itertools

class Storage:
    """
    Factory class for creating storage objects
    """
    
    def __new__(cls, config):

        if config['type'] == 'wandb':
            return WandbStorage(config)
        elif config['type'] == 'local':
            return LocalFileSystemStorage(config)
        elif config['type'] == 'ssh':
            return SSHFileSystemStorage(config)
        elif config['type'] in ['runs_db']:
            return RunsDBStorage(config)
        elif config['type'] in ['queue_db']:
            return QueueDBStorage(config)
        else:
            raise ValueError(f"Invalid storage type {config['type']}")


class BaseStorage(ABC):
    """
    Base Storage class
    """
    def __init__(self, config):
        self.config = config
        self.project = config.get('project')
        self.run = config.get('run')
        self.sweep = config.get('sweep')

    def save(self, filename, data, filetype, write_mode='w'):
        raise NotImplementedError
    
    def download(self, filename, directory, extract_archives=True):
        raise NotImplementedError
    
    def load(self, filename, filetype):
        """
        Download file into a temp dir and return object 
        Valid filetypes: 
                        - text: open the file as text
                        - torch: use torch.load to open file
                        - numpy: use numpy.load to open file           
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            
            dirname = os.path.dirname(filename)
            if len(dirname) > 0:
                # If dirname provided, make dir in tmp_dir
                tmp_dir = os.path.join(tmp_dir, dirname)
                os.makedirs(tmp_dir, exist_ok=True)
            # Download file to local machine
            self.download(filename, tmp_dir)

            if len(dirname) > 0:
                # TODO: Find a better solution
                #      when len(dirname) > 0, the file gets saved as tmp_dir/<basename>
                filename = os.path.basename(filename)

            # Load contents of downloaded file
            if filetype == 'text':
                with open(os.path.join(tmp_dir, filename), 'r') as f:
                    return f.read()
            elif filetype == 'torch':
                return torch.load(os.path.join(tmp_dir, filename))
            elif filetype == 'numpy':
                return np.load(os.path.join(tmp_dir, filename))
            elif filetype == 'env_yaml':
                return EnvYAML(os.path.join(tmp_dir, filename))
            elif filetype == 'yaml':
                with open(os.path.join(tmp_dir, filename), 'r') as f:
                    return yaml.safe_load(f)
            elif filetype == 'bytesio':
                with open(os.path.join(tmp_dir, filename), 'rb') as f:
                    return io.BytesIO(f.read())
            elif filetype == 'mp4':
                import skvideo.io
                return skvideo.io.vread(os.path.join(tmp_dir, filename))
            elif filetype == 'json':
                with open(os.path.join(tmp_dir, filename), 'r') as f:
                    return json.load(f)
                # return json.load(os.path.join(tmp_dir, filename))
            else:
                raise ValueError(f"Invalid filetype {filetype}")


    def load_from_archive(self, archive_name, filenames, filetypes):
        """
        Extract contents of the archive and return the files inside it
        The names of files (list) should match the filetypes (also list)
        """

        if not isinstance(filenames, list):
            filenames = [filenames]
            filetypes = [filetypes]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download and extract the archive
            self.download(archive_name, tmp_dir, extract_archives=True)
            outputs = {}
            for filename, filetype in zip(filenames, filetypes):
                # Load contents of downloaded file
                if filetype == 'text':
                    with open(os.path.join(tmp_dir, filename), 'r') as f:
                        outputs[filename] = f.read()
                elif filetype == 'torch':
                    outputs[filename] = torch.load(os.path.join(tmp_dir, filename))
                elif filetype == 'numpy':
                    outputs[filename] = np.load(os.path.join(tmp_dir, filename))
                elif filetype == 'env_yaml':
                    outputs[filename] = EnvYAML(os.path.join(tmp_dir, filename))
                elif filetype == 'yaml':
                    with open(os.path.join(tmp_dir, filename), 'r') as f:
                        outputs[filename] = yaml.safe_load(f)
                elif filetype == 'mp4':
                    import skvideo.io
                    outputs[filename] = skvideo.io.vread(os.path.join(tmp_dir, filename))  
                elif filetype == 'bytesio':
                    with open(os.path.join(tmp_dir, filename), 'rb') as f:
                         outputs[filename] = io.BytesIO(f.read())
                else:
                    raise ValueError(f"Invalid filetype {filetype}")
        return outputs
    
    def close_connection(self):
        pass

class WandbStorage(BaseStorage):
    """"
    Storage class for i/o from wandb
    """

    def __init__(self,config):
        super().__init__(config)

    def save(self, filename, object):
        raise NotImplementedError

    def download(self, filename, directory, extract_archives=True):
        """
        Download a file from wandb in the 'files' folder
        """
        api = wandb.Api()
        run = api.run(f"{self.project}/{self.run}")
        # Download the file into the directory dir
        run.file(filename).download(replace=True, root=directory)

        if filename.endswith('.zip') and extract_archives:
            # Unzip and save unzipped files in dir
            with zipfile.ZipFile(os.path.join(directory, filename), 'r') as zip_ref:
                zip_ref.extractall(directory)
        else:
            raise ValueError(f"Invalid filename {filename}")

    def upload(self, directory=None, files=None, new_dir=None):
        # Copy a directory or file to storage
        raise NotImplementedError

class LocalFileSystemStorage(BaseStorage):
    """
    Storage class for i/o from local file system
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.root_dir = config['root_dir']
        self.sub_dir = config.get('sub_dir')

        if self.project is None:
            self.dir = os.path.join(self.root_dir)
        else:
            if self.run is None:
                self.dir = os.path.join(self.root_dir, self.project)
            else:
                # /root/project/sweep/run or /root/project/run
                if self.sweep is None:
                    self.dir = os.path.join(self.root_dir, self.project, self.run)
                else:
                    self.dir = os.path.join(self.root_dir, self.project, f"sweep_{self.sweep}", self.run)

        if self.sub_dir:
            # /root/project/run/sub_dir
            self.dir = os.path.join(self.dir, self.sub_dir)
        
        if config.get('overwrite', False):
            if os.path.exists(self.dir):
                # Delete existing directory
                shutil.rmtree(self.dir)
        
        os.makedirs(self.dir, exist_ok=True)

    def save(self, filename, data, filetype, write_mode='w'):
        """
        Save data to filename (on remote machine) as filetype object
        """
        self.makedirs(os.path.dirname(filename))
        storage_filename = self.storage_path(filename)      
        # If filename specifices a non-existent director, create it
        

        if filetype == 'pickle':
            with open(storage_filename, mode='wb') as f:
                pickle.dump(data, f)
        elif filetype == 'text':
            with open(storage_filename, mode=write_mode) as f:
                f.write(data)
        elif filetype == 'torch':
            with open(storage_filename, mode='wb') as f:
                torch.save(data, f)
        elif filetype in ['yaml', 'env_yaml']:
            with open(storage_filename, mode='w') as f:
                yaml.safe_dump(data, f)
        elif filetype in ['mp4', 'gif']:
            with open(storage_filename, mode='wb') as f:
                assert isinstance(data, BytesIO)
                f.write(data.getvalue())
        elif filetype in ['json']:
            with open(storage_filename, mode='wb') as f:
                if isinstance(data, str):
                    data = json.loads(data)
                json.dump(data, f)
        else:
            raise NotImplementedError(f"Filetype {filetype} not implemented") 

    def download(self, filename, directory, extract_archives=True, delete_archive_after_extract=False):

        new_file = f"{directory}/{os.path.basename(filename)}"
        new_dir = os.path.dirname(new_file)
        if len(new_dir) > 0:
            os.makedirs(new_dir, exist_ok=True)

        # Copy file to dir
        shutil.copy(self.storage_path(filename), new_file)

        if filename.endswith('.zip') and extract_archives:
            # Unzip and save unzipped files in dir
            with zipfile.ZipFile(new_file, 'r') as zip_ref:
                zip_ref.extractall(new_dir)
                
            if delete_archive_after_extract:
                os.remove(new_file)

    def storage_path(self, filename):
        """
        return "self.dir/filename")
        """
        return os.path.join(self.dir, filename)

    def delete(self, directory=None, files=None):
        """
        Delete the directory
        """
        if directory is not None:
            # Delete specific folder
            shutil.rmtree(self.storage_path(directory))
        elif files is not None:
            # Delete files
            [os.remove(self.storage_path(f)) for f in files]
        else:
            # Delete the entire storage directory
            shutil.rmtree(self.dir)

    def makedirs(self, directory, exists_ok=True):
        """
        Create the directory
        """
        os.makedirs(self.storage_path(directory), exist_ok=True)

    def make_archive(self, files, zipfile_name):
        """
        Create a (zip) archive consistinng of files
        """
        # If zipfile_name has a non-existent dir, create it
        self.makedirs(os.path.dirname(zipfile_name))
        if not zipfile_name.endswith('.zip'):
            zipfile_name += '.zip'
        zip_filename = self.storage_path(zipfile_name)
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                # Determine the file name (without the path)
                file_name = self.storage_path(file_path).split('/')[-1]
                # Add the file to the zip file
                zipf.write(self.storage_path(file_path), file_name)

    def upload(self, directory=None, files=None, new_dir=None):
        # Copy a directory or file to storage
        dir = self.dir if new_dir is None else self.storage_path(new_dir)
        
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=False)
        if directory is not None:
            # Upload a directory to storage
            raise NotImplementedError
        elif files is not None:
            # Upload a list of files to storage
            if isinstance (files, str):
                files = [files]
            # with self.connection.open_sftp() as sftp:
            for file in files:
                file_name = file.split(f"{os.path.dirname(file)}/")[-1]
                shutil.copy(file, f"{dir}/{file_name}")
    
    def get_filenames(self, dir=None):
        """
        Return a list of filenames in the directory
        """
        check_dir = self.dir
        if dir is not None:
            check_dir = os.path.join(self.dir, dir)

        return glob.glob(check_dir)

    def archive_filenames(self, archive_name):
        try:
            with zipfile.ZipFile(self.storage_path(archive_name), 'r') as zip_file:
                return zip_file.namelist()
        except:
            return None
        

    def copy(self, file_or_dir, new_file_or_dir=None, mode='file'):
        """
        Copy file_or_dir to new_dir
        """
        if new_file_or_dir is None:
            new_dir = self.dir

        if mode == 'dir':
            # Copy a directory
            shutil.copytree(self.storage_path(file_or_dir), self.storage_path(new_dir))
        elif mode == 'file':
        # if mode == 'file':
            old_file = file_or_dir
            new_file_name = new_file_or_dir
            if new_file_name is None:
                new_file_name, extension = os.path.basename(old_file).split('.')
                new_file_name = f"{new_file_name}_copy.{extension}"

            # Copy a file
            shutil.copy(self.storage_path(old_file), self.storage_path(new_file_name))
        else:
            raise ValueError(f"Invalid mode {mode}")


class SSHFileSystemStorage(BaseStorage):
    """
    Storage class for i/o from a remote filesystem via ssh
    """

    def __init__(self, config):
        super().__init__(config)

        self.connection = paramiko.client.SSHClient()
        logging.getLogger("paramiko").setLevel(logging.WARNING)
        self.connection.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.connection.connect(config['host'], 
                                username=config['username'], 
                                password=config['password'])

        self.root_dir = config['root_dir']
        self.sub_dir = config.get('sub_dir')

        assert self.connection_active(), "Could not establish SSH connection"

        if self.project is None:
            self.dir = self.root_dir
        else:
            if self.run is None:
                self.dir = os.path.join(self.root_dir, self.project)
            else:
                # /root/project/sweep/run or /root/project/run
                if self.sweep is None:
                    self.dir = os.path.join(self.root_dir, self.project, self.run)
                else:
                    self.dir = os.path.join(self.root_dir, self.project, f"sweep_{self.sweep}", self.run)

        if self.sub_dir:
            # /root/project/run/sub_dir
            self.dir = os.path.join(self.dir, self.sub_dir)
    
            
        self.makedirs(self.dir, config.get('overwrite', False))

    def connection_active(self):
        return self.connection.get_transport().is_active()
        
    def close_connection(self):
        return self.connection.close()

    def save(self, filename, data, filetype, write_mode='w'):
        """
        Save data to filename (on remote machine) as filetype object
        """
        with self.connection.open_sftp() as sftp:
            storage_filename = self.storage_path(filename)        
            if filetype == 'pickle':
                with sftp.file(storage_filename, mode='wb') as f:
                    pickle.dump(data, f)
            elif filetype == 'text':
                with sftp.file(storage_filename, mode=write_mode) as f:
                    f.write(data)
            elif filetype == 'torch':
                with sftp.file(storage_filename, mode='wb') as f:
                    torch.save(data, f)
            elif filetype in ['yaml', 'env_yaml']:
                with sftp.file(storage_filename, mode='wb') as f:
                    yaml.safe_dump(data, f)
            elif filetype in ['mp4', 'gif']:
                assert isinstance(data, BytesIO)
                with sftp.file(storage_filename, mode='wb') as f:
                    f.write(data.getvalue())
            elif filetype in ['json']:
                with sftp.file(storage_filename, mode='wb') as f:
                    if isinstance(data, str):
                        data = json.loads(data)
                    json.dump(data, f)
            else:
                raise NotImplementedError(f"Filetype {filetype} not implemented")   

    def download(self, filename, directory, extract_archives=True, delete_archive_after_extract=False):
        """
        Download filename (must be basename) to dir (on local machine)
        """
        new_file = f"{directory}/{os.path.basename(filename)}"
        new_dir = os.path.dirname(new_file)

        if len(new_dir) > 0:
            os.makedirs(new_dir, exist_ok=True)
        with self.connection.open_sftp() as sftp:
            sftp.get(self.storage_path(filename), new_file)
        
        if filename.endswith('.zip') and extract_archives:
            # Unzip and save unzipped files in dir
            with zipfile.ZipFile(new_file, 'r') as zip_ref:
                zip_ref.extractall(directory)
            if delete_archive_after_extract:
                os.remove(new_file)


    def download_dir(self, remote_dirname, local_dirname):
        """
        Download the entire directory
        See: https://stackoverflow.com/questions/6674862/recursive-directory-download-with-paramiko
        """
        with self.connection.open_sftp() as sftp:
            self._recursive_sftp_get(sftp, remote_dirname, local_dirname)


    def _recursive_sftp_get(self, sftp, remote_dir, local_dir):
        """
        See: https://stackoverflow.com/questions/6674862/recursive-directory-download-with-paramiko
        """
        for entry in sftp.listdir_attr(remote_dir):
            remotepath = remote_dir + "/" + entry.filename
            localpath = os.path.join(local_dir, entry.filename)
            mode = entry.st_mode

            if self._isdir(sftp, mode=mode):
                try:
                    os.mkdir(localpath)
                except OSError:     
                    pass
                self._recursive_sftp_get(sftp, remotepath, localpath)
            else:
                sftp.get(remotepath, localpath)

    def _isdir(self, sftp, path=None, mode=None):
        if path is None:
            assert mode is not None
        else:
            mode = sftp.stat(path).st_mode

        try:
            return S_ISDIR(mode)
        except IOError:
            #Path does not exist, so by definition not a directory
            return False


    def storage_path(self, filename):
        """
        return "self.dir/filename"
        """
        return os.path.join(self.dir, filename)

    def makedirs(self, directory, overwrite=False):
        """
        If dir does not exist, create it
        creates dirs recursively e.g. if directory='a/b/c' and 
        'a' and 'a/b' do not exist, creates 'a', 'a/b', 'a/b/c'
        """
        # dir_exists = self.connection.run(f'cd {dir}', warn=True, hide=True)
        _, stdout, stderr = self.connection.exec_command(f'cd {directory}')
        if overwrite:
            if stdout.channel.recv_exit_status() == 0:
                # Delete existing directory
                _, stdout, stderr = self.connection.exec_command(f'rm -r {directory}')
                if stdout.channel.recv_exit_status() != 0:
                    raise StorageDeleteError(f"Could not delete directory {directory}. Error: {stderr.readlines()}")
                    # raise ValueError(f"Could not delete directory {directory}. Error: {stderr.readlines()}")
        else:
            if stdout.channel.recv_exit_status() == 0:
                # Directory already exists, so exit
                return 0
            
        # Create directory
        _, stdout, stderr = self.connection.exec_command(f'mkdir -p {directory}')
        if stdout.channel.recv_exit_status() != 0:
            # Could not create directory
            raise ValueError(f"Could not create directory {directory}. Error: {stderr.readlines()}")

    def delete(self, directory=None, files=None):
        """
        Delete the directory
        """
        if directory is not None:
            # Delete specific folder
            _, stdout, stderr = self.connection.exec_command(f'rm -r {self.storage_path(directory)}')
            if stdout.channel.recv_exit_status() != 0:
                # Could not delete directory
                # raise ValueError(f"Could not delete directory {self.dir}. Error: {stderr.readlines()}")
                raise StorageDeleteError(f"Could not delete directory {directory}. Error: {stderr.readlines()}")
        elif files is not None:
            # Delete files in the list
            raise NotImplementedError
        else:
            # TODO: rm -rf could be risky. Safer way?
            _, stdout, stderr = self.connection.exec_command(f'rm -r {self.dir}')
            if stdout.channel.recv_exit_status() != 0:
                # Could not delete directory
                raise StorageDeleteError(f"Could not delete directory {self.dir}. Error: {stderr.readlines()}")
                # raise ValueError(f"Could not delete directory {self.dir}. Error: {stderr.readlines()}")
        
    def make_archive(self, files):
        """
        Create a (zip) archive consistinng of files
        return path to archive
        """
        raise NotImplementedError


    def upload(self, directory=None, files=None, new_dir=None):

        remote_dir = self.dir if new_dir is None else self.storage_path(new_dir)
        self.makedirs(remote_dir)

        if directory is not None:
            # Upload a directory to storage
            raise NotImplementedError
        elif files is not None:
            # Upload a list of files to storage
            if isinstance (files, str):
                files = [files]
            with self.connection.open_sftp() as sftp:
                for file in files:
                    file_name = file.split(f"{os.path.dirname(file)}/")[-1]
                    sftp.put(file, f"{remote_dir}/{file_name}")
        else:
            raise NotImplementedError
        

    def get_filenames(self, dir=None):
        """
        Return a list of filenames in the directory
        """
        check_dir = self.dir
        if dir is not None:
            check_dir = os.path.join(self.dir, dir)
            
        _, stdout, stderr =  self.connection.exec_command(f'ls {check_dir}')
        filenames = stdout.readlines()
        filenames = [s.strip('\n') for s in filenames]
        return filenames
    
    def copy(self, file_or_dir, new_file_or_dir=None, mode='file'):
        """
        Copy file_or_dir to new_dir
        """
        if mode == 'file':
            old_file = file_or_dir
            new_file_name = new_file_or_dir
            if new_file_name is None:
                new_file_name, extension = os.path.basename(old_file).split('.')
                new_file_name = f"{new_file_name}_copy.{extension}"
            # Copy a file
            self.connection.exec_command(f'cp {self.storage_path(old_file)} {self.storage_path(new_file_name)}')
        elif mode == 'dir':
            raise NotImplementedError
        else:
            raise ValueError("mode must be 'dir' or 'file'")
        
    def archive_filenames(self, archive_name):
        _, stdout, stderr =  self.connection.exec_command(f"unzip -l {self.storage_path(archive_name)}")
        err = stderr.readlines()
        if len(err) > 0:
            # Could not unzip
            return None
        filenames = stdout.readlines()
        filenames = [x.split('\n')[0].split(' ')[-1] for x in filenames]
        filenames = filenames[3:-2]
        if len(filenames) < 1:
            # There were no files
            filenames = None
        return filenames

class DBStorage(BaseStorage):
    """
    Generic MYSQL DB Storage class
    """

    def __init__(self, config):
        # First, connect to the mysql server
        self.conn = mysql.connector.connect(
            host=config['host'],
            user=config['username'],
            password=config['password'],
        )

        databases = self.show_databases()
        if config['name'] not in databases:
            try:
                self.create_database(config['name'])
            except mysql.connector.errors.ProgrammingError:
                raise ValueError(f"Could not create database {config['name']}. Try creating database manually")

        # Reconnect -- this time to the database
        self.conn = mysql.connector.connect(
                    host=config['host'],
                    user=config['username'],
                    password=config['password'],
                    database=config['name']
                )
        
        assert self.conn.is_connected(), "Could not connect to database"

    def create_database(self, db_name):
        with self.conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {db_name}")
            self.conn.commit()


    def show_databases(self):
        with self.conn.cursor() as cursor:
            # Execute a query to retrieve a list of all databases
            cursor.execute("SHOW DATABASES")
            # Fetch the result
            databases = cursor.fetchall()
            # Extract the database names from the result
            return [db[0] for db in databases]

    def close_connection(self):
        return super().close_connection()

class RunsDBStorage(DBStorage):
    """
    DB Storage used for runs and sweeps (e.g. when evaluating)
    """

    def __init__(self, config):
        super().__init__(config)
        self._setup_runs_table()
        self._setup_metrics_info_table()

    def add_run(self, run_dict):
        """
        Add run_dict to the runs table; overwrite if it exists
        """
        COLUMNS = {'run_id', 'sweep', 'project', 'eval_name', 'value', 'value_std', 'step'}
        with self.conn.cursor() as cursor:
            cursor.execute(f"""INSERT INTO runs
                        (run_id, sweep, project, steps, folder)
                        VALUES ('{run_dict['run_id']}', 
                        '{run_dict['sweep']}', 
                        '{run_dict['project']}', 
                        '{run_dict['steps']}', 
                        '{run_dict['folder']}')
                        ON DUPLICATE KEY UPDATE
                        run_id='{run_dict['run_id']}', 
                        sweep='{run_dict['sweep']}', 
                        project='{run_dict['project']}',
                        steps='{run_dict['steps']}',
                        folder='{run_dict['folder']}'
                        """)

            self.conn.commit()

    def add_metric(self, metric_name, metric_dict, temporal=False):
        col_names = [*metric_dict['ids'].keys(), *metric_dict['data'].keys()]            
        ids = list(metric_dict['ids'].values())
        if temporal:
            # Add multiple rows
            rows = [(*ids,*x) for x in zip(*metric_dict['data'].values())]
            with self.conn.cursor() as cursor:
                insert_query = f"""INSERT INTO {metric_name}"""
                insert_query += f""" ({','.join([f"{x}" for x in col_names])})"""
                insert_query += f""" VALUES ({', '.join(['%s']*len(col_names))})"""
                duplicate_query = f"""ON DUPLICATE KEY UPDATE """
                duplicate_query += ', '.join([f"{col} = VALUES({col})" for col in col_names])
                insert_query = " ".join([insert_query, duplicate_query])
                cursor.executemany(insert_query,rows)
        else:
            # Add single row
            row = (*ids, *metric_dict['data'].values())
            with self.conn.cursor() as cursor:
                insert_query = f"""INSERT INTO {metric_name}"""
                insert_query += f""" ({', '.join(col_names)})"""
                insert_query += f""" VALUES ({', '.join(['%s']*len(col_names))})"""
                duplicate_query = f"""ON DUPLICATE KEY UPDATE """
                duplicate_query += ', '.join([f"{col} = VALUES({col})" for col in col_names])
                insert_query = " ".join([insert_query, duplicate_query])
                cursor.execute(insert_query,row)
            self.conn.commit()

    def add_metric_table(self, metric_name, metric):
        """
        If the database does not have a metric_name table, create it
        metric_table_spec: Optional specification for the metric table
                            if not provided, use the default specs
        """
        with self.conn.cursor() as cursor:
            # Check if table already exists
            cursor.execute(f"SHOW TABLES LIKE '{metric_name}'")
            metric_table_exists = cursor.fetchone() is not None
            if not metric_table_exists:
                # Create a new table
                cursor.execute(f"""CREATE TABLE {metric_name} ({metric.db_spec})""")
                # Add metric info to the metrics table
                cursor.execute(f"""
                               INSERT INTO metrics (name, type, temporal) 
                               VALUES ('{metric_name}', '{metric.type}', '{metric.temporal}')
                               ON DUPLICATE KEY UPDATE
                               name='{metric_name}', type='{metric.type}', temporal='{metric.temporal}'
                               """)

                self.conn.commit()

    def show_metric_table(self, metric_name, limit=100):
        """
        If there is a table 'metric_name' in the database, return it
        """
        with self.conn.cursor() as cursor:
            cursor.execute(f"SHOW TABLES LIKE '{metric_name}'")
            metric_table_exists = cursor.fetchone() is not None
            if metric_table_exists:
                query = f"SELECT * FROM {metric_name}"
                if limit is not None:
                    query += f" LIMIT {limit}"
                cursor.execute(query)
                data = cursor.fetchall()
                columns = [x[0] for x in cursor.description]
                data = pd.DataFrame(data, columns=columns)
                return data
            else:
                return None
            
    def _setup_metrics_info_table(self):
        """
        Table to store information about metrics
        """
        with self.conn.cursor() as cursor:
            cursor.execute(f"SHOW TABLES LIKE 'metrics'")
            runs_table_exists = cursor.fetchone()
            if runs_table_exists is None:
                cursor.execute(f"""CREATE TABLE metrics (
                            name VARCHAR(255),
                            type VARCHAR(255),
                            temporal VARCHAR(255),
                            PRIMARY KEY (name)
                )""")
                self.conn.commit()
    def _setup_runs_table(self):
        """
        If the database does not have a 'runs' table, create it
        """
        with self.conn.cursor() as cursor:
            cursor.execute(f"SHOW TABLES LIKE 'runs'")
            runs_table_exists = cursor.fetchone()
            if runs_table_exists is None:
                cursor.execute(f"""CREATE TABLE runs (
                            run_id VARCHAR(255),
                            sweep VARCHAR(255),
                            project VARCHAR(255),
                            steps VARCHAR(255),
                            folder VARCHAR(255),
                            PRIMARY KEY (run_id, sweep, project)
                )""")
                self.conn.commit()

    def show_runs(self, run_names=None, limit=None, output_format='pandas'):
        if run_names is not None:
            run_names = run_names if len(run_names) > 1 else [run_names]
            query = f"SELECT * FROM runs WHERE run_id IN ({run_names})"
        else:
            query = f"SELECT * FROM runs"

        if limit is not None:
            query += f" LIMIT {limit}"
        
        with self.conn.cursor() as cursor:
            cursor.execute(query)
            runs = cursor.fetchall()
            cursor.execute("DESCRIBE runs")
            runs_desc = cursor.fetchall()

        # Return a pandas dataframe?
        columns = [x[0] for x in runs_desc]
        if output_format == 'pandas':
            return pd.DataFrame(runs, columns=columns)
        elif output_format == 'dict':
            return {'columns': columns, 'data': runs}

        return runs


class StorageDeleteError(Exception):
    pass


class QueueDBStorage(DBStorage):
    
    def __init__(self, config):
        super().__init__(config)
        self.setup_queue_table(config.get('queue') or self.default_queue_table_config)

    @property
    def default_queue_table_config(self):
        return {
            'name': 'default_queue'
        }

    def setup_queue_table(self, config):
        """
        Current queue of runs
        """
        query = f"""
        CREATE TABLE IF NOT EXISTS {config['name']} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            project VARCHAR(255),
            exp VARCHAR(255),
            overrides JSON,
            run_id VARCHAR(255),
            exp_config JSON,
            study_config JSON,
            priority INT NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(255) DEFAULT 'Pending',
            heartbeat_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tags JSON,
            sweep_id VARCHAR(255),
            progress FLOAT
        );
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query)
            self.conn.commit()
        return 0
            
    def delete_queue_table(self, queue_name):
        """
        Delete a queue table
        """
        with self.conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE {queue_name}")
            self.conn.commit()
        return 0

    def enqueue_run(self, run_info):
        """
        Add a run to the queue
        """

        priority = run_info.get('priority') or 0 # Lowest priority
        study_config = run_info.get('study_config') or {}
        exp_config = run_info.get('exp_config') or {}
        overrides = run_info.get('overrides') or {}
        tags = run_info.get('tags') or {}
        queue_name = run_info['queue_name']

        with self.conn.cursor() as cursor:
            enqueue_query = f"""
            INSERT INTO {queue_name} (project, exp, overrides, run_id, exp_config, study_config, priority, tags, sweep_id, progress)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(enqueue_query, (
                run_info.get('project'),
                run_info.get('exp'),
                json.dumps(overrides),
                run_info.get('run_id'), 
                json.dumps(exp_config),
                json.dumps(study_config),
                priority,
                json.dumps(tags),
                run_info.get('sweep_id'),
                run_info.get('progress', 0.0)
                )
                )
            self.conn.commit()
        return 0

    def delete_run(self, run_info):
        """
        Delete a run from the queue
        """
        with self.conn.cursor() as cursor:
            query = f"DELETE FROM {run_info['queue_name']} WHERE id = %s"
            cursor.execute(query, (run_info.get('id'),))
            self.conn.commit()
        return 0

    def update_run(self, run_info, new_run_info):
        """
        Update a run in the queue
        """
        query = f"SELECT * FROM {run_info['queue_name']} WHERE id = %s;"
        run_db_id = int(run_info.get('id'))
        with self.conn.cursor(dictionary=True) as cursor:
            cursor.execute(query, (run_db_id,))
            run = cursor.fetchone()

            if run is None:
                raise ValueError(f"Run with id {run_db_id} not found")

            # Update the run
            update_query = f"UPDATE {run_info['queue_name']} SET "
            update_values = []
            for key, value in new_run_info.items():
                update_query += f"{key} = %s, "
                if value is not None:
                    if key in ['overrides', 'exp_config', 'study_config', 'tags']:
                        value = json.dumps(value)
                update_values.append(value)

            update_query = update_query.rstrip(", ") + " WHERE id = %s"
            update_values.append(run_db_id)
            cursor.execute(update_query, tuple(update_values))
            self.conn.commit()
            
        new_run_update = copy(run)
        new_run_update.update(new_run_info)
        return new_run_update


    def dequeue_run(self, queue_name, heartbeat_timeout=1):
        """
        Dequeue a run from the database. 
        heartbeat_timeout (seconds): 
            If the last heartbeat from the run came more than heartbeat_timeout seconds ago, 
            assume it is no longer running. So it can be returned.
        """
        with self.conn.cursor(dictionary=True) as cursor:

            # Check if table exists
            cursor.execute(f"SHOW TABLES LIKE '{queue_name}'")
            tables = cursor.fetchall()
            if len(tables) == 0:
                raise ValueError(f"Queue {queue_name} does not exist")

            # Wait for heartbeat
            with Progress() as progress:
                task = progress.add_task("[red] Heartbeat Timeout...", total=heartbeat_timeout)
                for _ in range(heartbeat_timeout):
                    progress.update(task, advance=1)
                    sleep(1)

            query = f"""SELECT *
            FROM {queue_name}
            WHERE status IS NULL OR status NOT IN ('Completed', 'Cancelled', 'Error', 'Paused')
            ORDER BY priority DESC
            LIMIT 100"""
            cursor.execute(query)
            runs = pd.DataFrame(cursor.fetchall())
            if len(runs) > 0:
                # Check if any runs are already running
                runs['heartbeat_at'] = runs['heartbeat_at'].apply(lambda x: x.replace(tzinfo=timezone.utc))
                runs['since_heartbeat'] = datetime.now(timezone.utc) - runs['heartbeat_at']
                runs['currently_running'] = (runs['status'] == 'InProgress') & (runs['since_heartbeat'] < timedelta(seconds=heartbeat_timeout))
                runs = runs[~runs['currently_running']]
            if len(runs) > 0:
                run = dict(runs.iloc[0])
                run['queue_name'] = queue_name
                return run
            
            return None

    def show_queue(self, queue_name):
        """
        Show the current queue
        """
        with self.conn.cursor(dictionary=True) as cursor:
            cursor.execute(f"SELECT * FROM {queue_name}")
            return cursor.fetchall()

    def show_queues(self):
        """
        Show all queues
        """
        with self.conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            tables = [x[0] for x in tables]
            return tables
        
    def add_runs_from_config(self, config):
        queue_name = config['queue_name']
        for run in  config['runs']:
            # Get overrides
            overrides = run.get('overrides') 
            overrides = dict(overrides) if overrides is not None else {}
            # Add queue to tag
            tags = run.get('tags') or []
            tags.append('queue_' + queue_name)
            run_info = {
                'project': run['project'],
                'exp': run['exp'],
                'overrides': overrides,
                'priority': run.get('priority', 0),
                'tags': tags,
                'queue_name': queue_name
            }
            if run.get('sweep'):
                run_info['sweep_id'] = run['sweep'].pop('id')
                if run_info['sweep_id'] is not None: 
                    run_info['tags'].append('sweep_' + run_info['sweep_id'])
                # TODO: Assumes each sweep has format {param: {values: [...]}}
                sweep_dict = {k:v['values'] for (k,v) in run['sweep'].items()}
                keys, values = zip(*sweep_dict.items())
                experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
                for exp in experiments:
                    new_run_info = deepcopy(run_info)
                    new_run_info['overrides'].update(exp)
                    self.enqueue_run(new_run_info)
            else:
                self.enqueue_run(run_info)
            
        return 0


