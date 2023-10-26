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

    @abstractmethod
    def save(self, filename, data, filetype, write_mode='w'):
        raise NotImplementedError
    
    @abstractmethod
    def download(self, filename, dir, extract_archives=True):
        raise NotImplementedError
    
    def load(self, filename, filetype):
        """
        Download file into a temp dir and return object 
        Valid filetypes: 
                        - text: open the file as text
                        - torch: use torch.load to open file
                        - numpy: use numpy.load to open file           
        """
        assert filetype in ['text', 'torch', 'numpy'], f"Invalid filetype {filetype}"
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
            else:
                raise ValueError(f"Invalid filetype {filetype}")


    def load_from_archive(self, archive_name, filenames, filetypes):
        """
        Extract contents of the archive and return the files inside it
        The names of files (list) should match the filetypes (also list)
        """

        single_file = False
        if not isinstance(filenames, list):
            filenames = [filenames]
            filetypes = [filetypes]
            single_file = True
        
        all_valid = all([x in ['text', 'numpy', 'torch'] for x in filetypes])
        assert all_valid, f"Invalid filetype in {filetypes}"
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download and extract the archive
            self.download(archive_name, tmp_dir, extract_archives=True)
            outputs = []
            for filename, filetype in zip(filenames, filetypes):
                # Load contents of downloaded file
                if filetype == 'text':
                    with open(os.path.join(tmp_dir, filename), 'r') as f:
                        outputs.append(f.read())
                elif filetype == 'torch':
                    outputs.append(torch.load(os.path.join(tmp_dir, filename)))
                elif filetype == 'numpy':
                    outputs.append(np.load(os.path.join(tmp_dir, filename)))
                else:
                    raise ValueError(f"Invalid filetype {filetype}")

        return outputs[0] if single_file else outputs

class WandbStorage(BaseStorage):
    """"
    Storage class for i/o from wandb
    """

    def __init__(self,config):
        super().__init__(config)

    def save(self, filename, object):
        raise NotImplementedError

    def download(self, filename, dir, extract_archives=True):
        """
        Download a file from wandb in the 'files' folder
        """
        api = wandb.Api()
        run = api.run(f"{self.project}/{self.run}")
        # Download the file into the directory dir
        run.file(filename).download(replace=True, root=dir)

        if filename.endswith('.zip') and extract_archives:
            # Unzip and save unzipped files in dir
            with zipfile.ZipFile(os.path.join(dir, filename), 'r') as zip_ref:
                zip_ref.extractall(dir)
        else:
            raise ValueError(f"Invalid filename {filename}")

    
class LocalFileSystemStorage(BaseStorage):
    """
    Storage class for i/o from local file system
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.root_dir = config['root_dir']
        self.sub_dir = config.get('sub_dir')
        if self.sub_dir:
            # dir is root/project/run/sub_dir
            self.dir = os.path.join(self.root_dir, self.project, self.run, self.sub_dir)
        else:
            # dir is root/project/run
            self.dir = os.path.join(self.root_dir, self.project, self.run)
        os.makedirs(self.dir, exist_ok=True)

    def save(self, filename, data, filetype, write_mode='w'):
        """
        Save data to filename (on remote machine) as filetype object
        """

        storage_filename = self.storage_path(filename)        
        if filetype == 'pickle':
            with open(storage_filename, mode='wb') as f:
                pickle.dump(data, f)
        elif filetype == 'text':
            with open(storage_filename, mode=write_mode) as f:
                f.write(data)
        elif filetype == 'torch':
            with open(storage_filename, mode='wb') as f:
                torch.save(data, f)
        else:
            raise NotImplementedError(f"Filetype {filetype} not implemented")  

    def download(self, filename, dir, extract_archives=True):
        # Copy file to dir
        shutil.copy(self.storage_path(filename), f"{dir}/{filename}")

        if filename.endswith('.zip') and extract_archives:
            # Unzip and save unzipped files in dir
            with zipfile.ZipFile(os.path.join(dir, filename), 'r') as zip_ref:
                zip_ref.extractall(dir)

    def storage_path(self, filename):
        """
        return "self.dir/filename")
        """
        return os.path.join(self.dir, filename)

    def delete(self):
        """
        Delete the directory
        """
        shutil.rmtree(self.dir)

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


        self.root_dir = config['root_dir']
        self.sub_dir = config.get('sub_dir')
        if self.sub_dir:
            # dir is root/project/run/sub_dir
            self.dir = os.path.join(self.root_dir, self.project, self.run, self.sub_dir)
        else:
            # dir is root/project/run
            self.dir = os.path.join(self.root_dir, self.project, self.run)
        self._makedirs(self.dir)

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
            else:
                raise NotImplementedError(f"Filetype {filetype} not implemented")   

    def download(self, filename, dir, extract_archives=True):
        """
        Download filename (must be basename) to dir (on local machine)
        """
        with self.connection.open_sftp() as sftp:
            sftp.get(self.storage_path(filename), f"{dir}/{filename}")
    
        if filename.endswith('.zip') and extract_archives:
            # Unzip and save unzipped files in dir
            with zipfile.ZipFile(os.path.join(dir, filename), 'r') as zip_ref:
                zip_ref.extractall(dir)
        
    def storage_path(self, filename):
        """
        return "self.dir/filename"
        """
        return os.path.join(self.dir, filename)

    def _makedirs(self, dir):
        """
        If dir does not exist, create it
        creates dirs recursively e.g. if dir='a/b/c' and 
        'a' and 'a/b' do not exist, creates 'a', 'a/b', 'a/b/c'
        """
        # dir_exists = self.connection.run(f'cd {dir}', warn=True, hide=True)
        _, stdout, stderr = self.connection.exec_command(f'cd {dir}')
        if stdout.channel.recv_exit_status() != 0:
            # Create folder if it does not exist
            _, stdout, stderr = self.connection.exec_command(f'mkdir -p {dir}')
            if stdout.channel.recv_exit_status() != 0:
                # Could not create directory
                raise ValueError(f"Could not create directory {dir}. Error: {stderr.readlines()}")

    def delete(self):
        """
        Delete the directory
        """
        _, stdout, stderr = self.connection.exec_command(f'cd {self.dir}')
        if stdout.channel.recv_exit_status() != 0:
            # Create folder if it does not exist
            # TODO: rm -rf could be risky. Safer way?
            _, stdout, stderr = self.connection.exec_command(f'rm -rf {self.dir}')
            if stdout.channel.recv_exit_status() != 0:
                # Could not create directory
                raise ValueError(f"Could not delete directory {self.dir}. Error: {stderr.readlines()}")
    

class DBStorage(BaseStorage):
    """
    Database storage for e.g. sweeps
    """
    def __init__(self, config):
        raise NotImplementedError("DBStorage not implemented yet")