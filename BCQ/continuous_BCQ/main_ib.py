import argparse
import gym
import numpy as np
import os
import torch

import BCQ
import DDPG
import utils

from datetime import datetime


# Handles interactions with the environment, i.e. train behavioral or generate buffer
def interact_with_environment(env, state_dim, action_dim, max_action, device, args):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"
	# buffer_name = f"replay_buffer_test"

	# Initialize and load policy
	policy = DDPG.DDPG(state_dim, action_dim, max_action, device)     #, args.discount, args.tau)
	if args.generate_buffer: policy.load(f"./models/behavioral_{setting}")

	# Initialize buffer
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
	
	evaluations = []

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# Select action with noise
		if (
			(args.generate_buffer and np.random.uniform(0, 1) < args.rand_action_p) or 
			(args.train_behavioral and t < args.start_timesteps)
		):
			action = env.action_space.sample()
		else: 
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.gaussian_std, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if args.train_behavioral and t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if args.train_behavioral and (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/behavioral_{setting}", evaluations)
			policy.save(f"./models/behavioral_{setting}")

	# Save final policy
	if args.train_behavioral:
		policy.save(f"./models/behavioral_{setting}")

# Save final buffer and performance
	else:
		evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/buffer_performance_{setting}", evaluations)
		replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(state_dim, action_dim, max_action, device, args):
	# For saving files
	# setting = f"{args.env}_{args.seed}"
	# buffer_name = f"{args.buffer_name}_{setting}"
	buffer_name = f"replay_buffer_ib"
	# Initialize policy
	policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

	# Load buffer
	replay_buffer_train = utils.ReplayBuffer(state_dim, action_dim, device)
	replay_buffer_train.load(f"./buffers/{buffer_name}_train")
	replay_buffer_eval = utils.ReplayBuffer(state_dim, action_dim, device)
	replay_buffer_eval.load(f"./buffers/{buffer_name}_eval")

	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0
	pol_trains_seq = {
		'vae_loss_seq': [],
		'critic_loss_seq': [],
		'actor_loss_seq': []
	}
	pol_evals_seq = {
		'vae_loss_seq': [],
		'critic_loss_seq': [],
		'actor_loss_seq': []
	}

	while training_iters < args.max_timesteps: 
		pol_trains = policy.train(replay_buffer_train, iterations=int(args.eval_freq), batch_size=args.batch_size)
		# 验证集
		pol_evals = policy.train(replay_buffer_eval, iterations=int(args.eval_freq / 10), batch_size=args.batch_size)

		# evaluations.append(eval_policy(policy, args.env, args.seed))

		pol_trains_seq['vae_loss_seq'].append(pol_trains['vae_loss_seq'].mean())
		pol_evals_seq['vae_loss_seq'].append(pol_evals['vae_loss_seq'].mean())
		pol_trains_seq['critic_loss_seq'].append(pol_trains['critic_loss_seq'].mean())
		pol_evals_seq['critic_loss_seq'].append(pol_evals['critic_loss_seq'].mean())
		pol_trains_seq['actor_loss_seq'].append(pol_trains['actor_loss_seq'].mean())
		pol_evals_seq['actor_loss_seq'].append(pol_evals['actor_loss_seq'].mean())
		# # plt 绘图
		# utils.loss_visualization(pol_trains, 'train', fig_path, training_iters)
		# print(f"train loss plt over----------{training_iters}")
		# utils.loss_visualization(pol_evals, 'eval', fig_path, training_iters)
		# print(f"eval loss plt over----------{training_iters}")

		# np.save(f"./results/BCQ_{setting}", evaluations)

		training_iters += args.eval_freq
		print(f"Training iterations: {training_iters}")

	# plt 绘图
	utils.loss_visualization(pol_trains_seq, 'train', fig_path, training_iters)
	print(f"train loss plt over----------{training_iters}")
	utils.loss_visualization(pol_evals_seq, 'eval', fig_path, training_iters)
	print(f"eval loss plt over----------{training_iters}")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# parser.add_argument("--env", default="HalfCheetah-v1")               # OpenAI gym environment name
	parser.add_argument("--env", default="ib")
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
	parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used before training behavioral
	parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
	parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
	parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
	parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
	parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
	parser.add_argument("--generate_buffer", action="store_true")   # If true, generate buffer
	parser.add_argument("--lr", default=1e-3, type=float)  			# lr for BCQ Network optimizer
	args = parser.parse_args()

	print("---------------------------------------")	
	if args.train_behavioral:
		print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
	elif args.generate_buffer:
		print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
	else:
		print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.train_behavioral and args.generate_buffer:
		print("Train_behavioral and generate_buffer cannot both be true.")
		exit()

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	fig_path = f"./results/{args.env}/fig/{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.lr}"
	if not os.path.exists(fig_path)	:
		os.makedirs(fig_path)


	# env = gym.make(args.env)
	#
	# env.seed(args.seed)
	# # env.action_space.seed(args.seed)
	# torch.manual_seed(args.seed)
	# np.random.seed(args.seed)
	#
	# state_dim = env.observation_space.shape[0]
	# action_dim = env.action_space.shape[0]
	# max_action = float(env.action_space.high[0])
	state_dim = 6
	action_dim = 3
	max_action = 2

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if args.train_behavioral or args.generate_buffer:
		interact_with_environment(env, state_dim, action_dim, max_action, device, args)
	else:
		train_BCQ(state_dim, action_dim, max_action, device, args)
