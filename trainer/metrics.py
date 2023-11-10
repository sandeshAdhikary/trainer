import numpy as np
from einops import rearrange
from io import BytesIO
import imageio
from PIL import Image
from copy import deepcopy
import string
import random 
import pandas as pd
import altair as alt
from trainer.utils import pretty_title, COLORS

ALPHABETS = list(string.ascii_lowercase)

class Metric:
    def __new__(cls, config):
        
        # If the metric is a built-in metric, return the corresponding class
        metric_name = config.get('name')
        if metric_name == 'avg_episode_reward':
            return AvgEpisodeReward(config)
        elif metric_name == 'episode_rewards':
            return EpisodeRewards(config)
        elif metric_name == 'observation_videos':
            return ObservationVideos(config)

        # Create a new metric based on type
        metric_type = config['type']
        temporal = config['temporal']
        if metric_type=='scalar':
            if temporal:
                return TemporalScalarMetric(config)
            else:
                return StaticScalarMetric(config)
        else:
            raise NotImplementedError()

class StaticMetric():
    """
    Single valued metrics i.e. metrics without steps
    """
    def __init__(self, config=None):
        self.temporal = False
        self.config = config or {}
    
    @property
    def db_spec(self):
        return """
                run_id VARCHAR(255),
                sweep VARCHAR(255),
                project VARCHAR(255),
                eval_name VARCHAR(255),
                value FLOAT,
                value_std FLOAT,
                PRIMARY KEY (run_id, sweep, eval_name)
            """

    def db_dict(self, ids, data):
        db_dict = {'ids': {}, 'data': {}}
        # Fill id columns
        for k, v in ids.items():
            if k in ['run_id', 'sweep', 'project', 'eval_name']:
                db_dict['ids'][k] = v
        # Fill data columns
        db_dict['data']['value'] = data['avg']
        db_dict['data']['value_std'] = data['std']
        return db_dict
    
    def _random_data(self):
        """
        Generate random data; useful in designing visualizations
        """
        num_runs = 5
        rows = []
        for idx in range(num_runs):
            run = {'run_id': ''.join(np.random.choice(ALPHABETS, size=5)),
                   'sweep': np.random.choice(['sweep_1', 'sweep_2']),
                   'project': np.random.choice(['project_1', 'project_2']),
                   'eval_name': np.random.choice(['eval1', 'eval2']),
                   'group': np.random.choice(['group1', 'group2', 'group3']),
                    'value': 10*np.random.random(),
                    'value_std': np.random.random(),
                   }
            rows.append(run)
        return pd.DataFrame(rows)

    def plot(self, data, facet=None, chart_properties=None, show_legend=True):
        # facet_by = 'eval_name'
        metric_name = self.__class__.__name__

        colors = COLORS['tableau20']
        data['value_high'] = data['value'] + data['value_std']
        data['value_low'] = data['value'] - data['value_std']
        max_value = data['value_high'].max()
        min_value = data['value_low'].min()


        group_names = list(data['group'].unique())
        
        color = alt.Color('group:N')
        if not show_legend:
            color = color.legend(None)

        bar_chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('group:N',  axis=alt.Axis(labels=False), title=""),
            y=alt.Y('value:Q', title=pretty_title(metric_name), scale=alt.Scale(domain=[min_value, max_value])),
            color=color.scale(domain=group_names, range=colors[:len(group_names)]),
        )

        line_high = alt.Chart(data).mark_errorbar(color='white').encode(
            alt.X('group:N',  axis=alt.Axis(labels=False), title=""),
            alt.Y('value_high:Q', title="",scale=alt.Scale(domain=[min_value, max_value])),
            alt.Y2('value_low:Q', title=""),
        )

        chart = bar_chart + line_high

        if chart_properties is not None:
            chart = chart.properties(**chart_properties)

        if facet is not None:
            chart = chart.facet(
                facet=facet['name'],
                columns=facet['columns']
            )

        chart_info = {
            'legend_colors': {k:v for k,v in zip(group_names, colors[:len(group_names)])}
        }

        return chart, chart_info


class TemporalMetric():
    """
    metrics with steps
    """
    def __init__(self, config=None):
        self.temporal = True
        self.config = config or {}

    @property
    def db_spec(self):
        return """
                run_id VARCHAR(255),
                sweep VARCHAR(255),
                project VARCHAR(255),
                eval_name VARCHAR(255),
                value FLOAT,
                value_std FLOAT,
                step INT,
                PRIMARY KEY (run_id, sweep, eval_name, step)
            """
          
    def db_dict(self, ids, data):
        db_dict = {'ids': {}, 'data': {}}
        # Fill id columns
        for k, v in ids.items():
            if k in ['run_id', 'sweep', 'project', 'eval_name']:
                db_dict['ids'][k] = v
        # Fill data columns
        db_dict['data']['value'] = data['avg']
        db_dict['data']['value_std'] = data['std']
        db_dict['data']['step'] = list(range(len(data['avg'])))
        return db_dict
        

    def plot(self, data, facet=None, chart_properties=None, show_legend=True):
        colors = COLORS['tableau20']

        
        metric_name = self.__class__.__name__

        data['value_high'] = data['value'] + data['value_std']
        data['value_low'] = data['value'] - data['value_std']
        max_value = data['value_high'].max()
        min_value = data['value_low'].min()
        
        
        # chart_title = alt.TitleParams(
        #     f"{pretty_title(metric)} for {pretty_title(eval_name)}",
        #     subtitle=["Error bands correspond to standard deviation across evaluation seeds"],
        #     subtitleColor='white'
        # )
        group_names = list(data['group'].unique())
        chart_title = "Test"


        color = alt.Color('group:N')
        if not show_legend:
            color = color.legend(None)

        chart = alt.Chart(data, title=chart_title).mark_line().encode(
            x=alt.X('step:Q'),
            y=alt.Y('value:Q', title=pretty_title(metric_name), scale=alt.Scale(domain=[min_value, max_value])),
            color=color.scale(domain=group_names, range=colors[:len(group_names)]),
        )

        error_band = alt.Chart(data).mark_errorband(opacity=0.5).encode(
            alt.Y('value_high:Q', title='', scale=alt.Scale(domain=[min_value, max_value])),
            alt.Y2('value_low:Q', title=''),
            alt.X('step:Q'),
            color=color
        )
        chart = chart + error_band

        if chart_properties is not None:
            chart = chart.properties(**chart_properties)
        
        if facet is not None:
            chart = chart.facet(
                facet=facet['name'],
                columns=facet['columns']
            )
        chart_info = {
            'legend_colors': {k:v for k,v in zip(group_names, colors[:len(group_names)])}
        }

        return chart, chart_info

    def _random_data(self):
        """
        Generate random data; useful in designing visualizations
        """
        num_runs = 5
        num_steps = 10
        rows = []
        for idx in range(num_runs):
            run = {'run_id': ''.join(np.random.choice(ALPHABETS, size=5)),
                   'sweep': np.random.choice(['sweep_1', 'sweep_2']),
                   'project': np.random.choice(['project_1', 'project_2']),
                   'eval_name': np.random.choice(['eval1', 'eval2']),
                   'group': np.random.choice(['group1', 'group2', 'group3'])
                   }
            for step in range(num_steps):
                run_step = deepcopy(run)
                run_step.update({
                    'step': step,
                    'value': 10*np.random.random(),
                    'value_std': np.random.random(),
                    })
                rows.append(run_step)
        return pd.DataFrame(rows)


class StaticScalarMetric(StaticMetric):
    """
    Single valued scalar metrics
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.type = 'scalar'
        

class TemporalScalarMetric(TemporalMetric):
    """
    Multi step scalar metrics
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.type = 'scalar'



class StaticObjectMetric(StaticMetric):
    """"
    For metrics that are objects (images, videos, etc)
    The objects are saved to file, 
    so the database entry will include the filepath
    """
    def __init__(self, config=None):
        super().__init__(config)

    @property
    def db_spec(self):
        return """
                run_id VARCHAR(255),
                sweep VARCHAR(255),
                project VARCHAR(255),
                eval_name VARCHAR(255),
                filepath VARCHAR(255),
                PRIMARY KEY (run_id, sweep, eval_name)
            """


class ImageMetric(StaticObjectMetric):
    """
    Single step images
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.type = 'image'


class VideoMetric(StaticObjectMetric):
    """
    Single step videos.
    We assume data is provided as frames or array of images
    They will be converted to and saved as a single video file
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.type = 'video'
        self.frame_rate = self.config.get('frame_rate', 20)
        self.quality = self.config.get('quality', 10)
        self.quality = min(10, self.quality)
        self.macro_block_size = 16 # ffmpeg writer's default
        self.img_size = (112,112)
    
    def db_dict(self, ids, data):
        db_dict = {'ids': {}, 'data': {}}
        # Fill id columns
        for k, v in ids.items():
            if k in ['run_id', 'sweep', 'project', 'eval_name']:
                db_dict['ids'][k] = v
        # Fill data columns
        db_dict['data']['filepath'] = data['filepath']
        return db_dict

    
    def imgs_to_video(self, imgs, video_format='mp4'):
        # Resize images so they are small and are divisible by ffmpeg's macro_block_size
        video = BytesIO()
        # Create a video writer using FFmpeg
        with imageio.get_writer(video, format=video_format, fps=self.frame_rate,
                                codec='h264',
                                quality=self.quality) as video_writer:
            for frame in imgs:
                # old_size = frame.shape[:-1]
                # new_size = (old_size[0] - (old_size[0]%self.macro_block_size),
                            # old_size[1] - (old_size[1]%self.macro_block_size))
                frame = Image.fromarray(frame)
                frame = frame.resize(self.img_size)
                # Convert the NumPy array to an image
                image = imageio.core.util.Array(np.array(frame))
                video_writer.append_data(image)
        return video
    
    def plot(self, data, facet=None, chart_properties=None, show_legend=True):
        pass


####################

class ObservationVideos(VideoMetric):
    def __init__(self, config=None):
        super().__init__(config)
        self.num_channels = self.config.get('num_channels', 3)
        self.max_frames = self.config.get('max_frames', 3)
        self.max_envs = self.config.get('max_envs', 3)
        self.video_format = self.config.get('video_format', 'mp4')


    def log(self, eval_output, storage, filename):
        obses = eval_output['obses'] # (T, num_envs, num_frames*C, H, W)
        obses = rearrange(obses, 't e (f c) h w -> t e f c h w', c=self.num_channels)
        # Select out max num envs and frames
        obses = obses[:,:self.max_envs,:self.max_frames,:]  
        # Stack frames horizontally and environments vertically
        obses = rearrange(obses, 't e f c h w -> t (e h) (f w) c')
        video_bytesio = self.imgs_to_video(obses, self.video_format)
        storage.save(f"{filename}.{self.video_format}", video_bytesio, filetype=self.video_format)
        return {'filepath': storage.storage_path(f"{filename}.{self.video_format}")}



class AvgEpisodeReward(StaticScalarMetric):
    def log(self, eval_output):
        rews = eval_output['rewards'] # (T, num_envs)
        rews = np.nanmean(rews, axis=0) # Mean across timesteps
        return {
            'avg': rews.mean(), # Mean across envs
            'std': rews.std() # Std across envs
        }
        
class EpisodeRewards(TemporalScalarMetric):

    def __init__(self, config=None):
        super().__init__(config)

    def log(self, eval_output):
        """
        return episode_rewards from eval_output
        #TODO: Need to account for nans?
        """
        rews = eval_output['rewards'] # (T, num_envs)
        return {
            'avg':  np.nanmean(rews, axis=1), # Mean across envs,
            'std': np.nanstd(rews, axis=1) # Std across envs
        }
    
