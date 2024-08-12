import gymnasium as gym
from trainer.rl.env_libraries.dm_control.make_dmc2gym_env import make_dmc_env

def make_env(args):
    if args['library'] == 'gym':
        return make_gym_env(args['env_id'], args.get('max_episode_steps'))
    if args['library'] == 'dmc2gym':
        return make_dmc_env(
            args['domain_name'],
            args['task_name'],
            resource_files=args.get('resource_files'),
            total_frames=args.get('total_frames'),
            seed=args.get('seed', 123),
            visualize_reward=args.get('visualize_reward', False),
            from_pixels=args.get('from_pixels', False),
            height=args.get('height', 84),
            width=args.get('width', 84),
            frame_skip=args.get('frame_skip', 1),
            episode_length=args.get('episode_length', 1000),
            environment_kwargs=args.get('environment_kwargs'),
        )


def make_gym_env(
        env_id,
        max_episode_steps=None,
        ):
    return gym.make(env_id, max_episode_steps=max_episode_steps, render_mode='rgb_array')
    