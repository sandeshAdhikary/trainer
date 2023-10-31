from abc import ABC, abstractmethod
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
        else:
            raise ValueError(f"Invalid storage type {config['type']}")


class BaseStorage(ABC):
    """
    Base Storage class
    """
    def __init__(self, config):
        self.config = config
        self.project = config['project']
        self.run = config['run']
        self.sweep = config.get('sweep')

    @abstractmethod
    def save(self, filename, data, filetype, write_mode='w'):
        raise NotImplementedError
    
    @abstractmethod
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
            # Download file to local machine
            self.download(filename, tmp_dir)
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
                else:
                    raise ValueError(f"Invalid filetype {filetype}")
        return outputs

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

        # /root/project/sweep/run or /root/project/run
        if self.sweep is None:
            self.dir = os.path.join(self.root_dir, self.project, self.run)
        else:
            self.dir = os.path.join(self.root_dir, self.project, f"sweep_{self.sweep}", self.run)

        if self.sub_dir:
            # /root/project/run/sub_dir
            self.dir = os.path.join(self.dir, self.sub_dir)
            
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
        else:
            raise NotImplementedError(f"Filetype {filetype} not implemented")  

    def download(self, filename, directory, extract_archives=True):
        # Copy file to dir
        shutil.copy(self.storage_path(filename), f"{directory}/{filename}")

        if filename.endswith('.zip') and extract_archives:
            # Unzip and save unzipped files in dir
            with zipfile.ZipFile(os.path.join(directory, filename), 'r') as zip_ref:
                zip_ref.extractall(directory)

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
        raise NotImplementedError

class SSHFileSystemStorage(BaseStorage):
    """
    Storage class for i/o from a remote filesystem via ssh
    """

    def __init__(self, config):
        super().__init__(config)

        self.connection = paramiko.client.SSHClient()
        self.connection.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.connection.connect(config['host'], 
                                username=config['username'], 
                                password=config['password'])

        # Check the SSH connection
        ssh_active = self.connection.get_transport().is_active()
        assert ssh_active, "Could not establish SSH connection"

        self.root_dir = config['root_dir']
        self.sub_dir = config.get('sub_dir')

        # /root/project/sweep/run or /root/project/run
        if self.sweep is None:
            self.dir = os.path.join(self.root_dir, self.project, self.run)
        else:
            self.dir = os.path.join(self.root_dir, self.project, f"sweep_{self.sweep}", self.run)

        if self.sub_dir:
            # /root/project/run/sub_dir
            self.dir = os.path.join(self.dir, self.sub_dir)
    
            
        self.makedirs(self.dir)

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
            else:
                raise NotImplementedError(f"Filetype {filetype} not implemented")   

    def download(self, filename, directory, extract_archives=True):
        """
        Download filename (must be basename) to dir (on local machine)
        """
        with self.connection.open_sftp() as sftp:
            sftp.get(self.storage_path(filename), f"{directory}/{filename}")
    
        if filename.endswith('.zip') and extract_archives:
            # Unzip and save unzipped files in dir
            with zipfile.ZipFile(os.path.join(directory, filename), 'r') as zip_ref:
                zip_ref.extractall(directory)
        
    def storage_path(self, filename):
        """
        return "self.dir/filename"
        """
        return os.path.join(self.dir, filename)

    def makedirs(self, directory):
        """
        If dir does not exist, create it
        creates dirs recursively e.g. if directory='a/b/c' and 
        'a' and 'a/b' do not exist, creates 'a', 'a/b', 'a/b/c'
        """
        # dir_exists = self.connection.run(f'cd {dir}', warn=True, hide=True)
        _, stdout, stderr = self.connection.exec_command(f'cd {directory}')
        if stdout.channel.recv_exit_status() != 0:
            # Create folder if it does not exist
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
            raise NotImplementedError
        elif files is not None:
            # Delete files in the list
            raise NotImplementedError
        else:
            # Delete the entire storage directory
            _, stdout, stderr = self.connection.exec_command(f'cd {self.dir}')
            if stdout.channel.recv_exit_status() != 0:
                # Create folder if it does not exist
                # TODO: rm -rf could be risky. Safer way?
                _, stdout, stderr = self.connection.exec_command(f'rm -rf {self.dir}')
                if stdout.channel.recv_exit_status() != 0:
                    # Could not create directory
                    raise ValueError(f"Could not delete directory {self.dir}. Error: {stderr.readlines()}")
        
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
        

class DBStorage(BaseStorage):
    """
    Database storage for e.g. sweeps
    """
    def __init__(self, config):
        raise NotImplementedError("DBStorage not implemented yet")