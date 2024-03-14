import gym

def make_env(args):
    if args.library == 'gym':
        return make_gym_env(args.env_id, args.max_episode_steps)
    if args.library == 'dmc2gym':
        pass


def make_gym_env(
        env_id,
        max_episode_steps=None,
        ):
    return gym.make(env_id, max_episode_steps=max_episode_steps)
    