from prey_env import gym_Environment_D as Environment
import time
def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  ## For non-image
  config = config.update({
      'logdir': f'logdir/{int(time.time())}',  # this was just changed to generate a new log dir every time for testing
      'run.train_ratio': 64,
      'run.log_every': 30,
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': '.*',
      'decoder.mlp_keys': '.*',
      'encoder.cnn_keys': '$^',
      'decoder.cnn_keys': '$^',
      'jax.platform': 'cpu',  # I don't have a gpu locally
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])


  from embodied.envs import from_gym
  env = Environment(e=2, predator_speed=0.2, has_predator=True, max_step=300)
  env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.'image'
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  #embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
