import gymnasium as gym
from gymnasium.envs.registration import register

def make_dmc_env(
    domain_name,
    task_name,
    resource_files=None,
    img_source=None,
    total_frames=1000,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1_000,
    environment_kwargs=None
):
    
    env_id = 'dmc_%s_%s_%s_%s-v1' % (domain_name, task_name, seed, img_source or 'none')

    if from_pixels:
        # No allowed for pixels since the visualization changes observation colors based on reward
        assert not visualize_reward, 'cannot use visualize reward when learning from pixels'

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if not env_id in gym.envs.registry:
        register(
            id=env_id,
            entry_point='trainer.rl.env_libraries.dm_control.dmc2gym.wrappers:DMCWrapper',
            kwargs={
                'domain_name': domain_name,
                'task_name': task_name,
                'resource_files': resource_files,
                'img_source': img_source,
                'total_frames': total_frames,
                'task_kwargs': {
                    'random': seed
                },
                'environment_kwargs': environment_kwargs,
                'visualize_reward': visualize_reward,
                'from_pixels': from_pixels,
                'height': height,
                'width': width,
                'camera_id': camera_id,
                'frame_skip': frame_skip,
            }
            ,
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)
