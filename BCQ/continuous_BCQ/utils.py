import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.size])
		np.save(f"{save_folder}_action.npy", self.action[:self.size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)

	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")

		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]


def loss_visualization(outputs, text, fig_path, training_iters):
	from matplotlib import pyplot as plt
	import os

	fig_path = fig_path
	plt_label = text
	y_vae_loss = outputs['vae_loss_seq']
	y_critic_loss = outputs['critic_loss_seq']
	y_actor_loss = outputs['actor_loss_seq']
	x_data_step = [x for x in range(len(y_vae_loss))]

	# best_epoch = y_dataset_eval_rrse.index(min(y_dataset_eval_rrse))

	# vae_loss
	plt.figure()
	plt.plot(x_data_step, y_vae_loss, color='red', label="vae_loss")
	# plt.axvline(best_epoch, color='r', linestyle='--', label='best epoch')
	plt.xlabel('Step')
	plt.ylabel('loss')
	plt.title(f'{plt_label}_vae_loss_{training_iters}')
	plt.legend()
	plt.savefig(os.path.join(fig_path, f'{plt_label}_vae_loss_{training_iters}.png'))
	plt.close()  # 关闭图像，避免出现warning

	# critic_loss
	plt.figure()
	plt.plot(x_data_step, y_critic_loss, color='blue', label="critic_loss")
	# plt.axvline(best_epoch, color='r', linestyle='--', label='best epoch')
	plt.xlabel('Step')
	plt.ylabel('loss')
	plt.title(f'{plt_label}_critic_loss_{training_iters}')
	plt.legend()
	plt.savefig(os.path.join(fig_path, f'{plt_label}_critic_loss_{training_iters}.png'))
	plt.close()  # 关闭图像，避免出现warning

	# actor_loss
	plt.figure()
	plt.plot(x_data_step, y_actor_loss, color='black', label="actor_loss")
	# plt.axvline(best_epoch, color='r', linestyle='--', label='best epoch')
	plt.xlabel('Step')
	plt.ylabel('loss')
	plt.title(f'{plt_label}_actor_loss_{training_iters}')
	plt.legend()
	plt.savefig(os.path.join(fig_path, f'{plt_label}_actor_loss_{training_iters}.png'))
	plt.close()  # 关闭图像，避免出现warning

	# all
	plt.figure()
	plt.plot(x_data_step, y_vae_loss, color='red', label="vae_loss")
	plt.plot(x_data_step, y_critic_loss, color='blue', label="critic_loss")
	plt.plot(x_data_step, y_actor_loss, color='black', label="actor_loss")
	# plt.axvline(best_epoch, color='r', linestyle='--', label='best epoch')
	plt.xlabel('Step')
	plt.ylabel('loss')
	plt.title(f'{plt_label}_loss_{training_iters}')
	plt.legend()
	plt.savefig(os.path.join(fig_path, f'{plt_label}_loss_{training_iters}.png'))
	plt.close()  # 关闭图像，避免出现warning


def reward_visualization(reward_seq, fig_path, training_iters):
	from matplotlib import pyplot as plt
	import os

	fig_path = fig_path
	plt_label = 'reward'
	reward_seq = reward_seq
	x_data_step = [x for x in range(len(reward_seq))]

	# best_epoch = y_dataset_eval_rrse.index(min(y_dataset_eval_rrse))

	plt.figure()
	plt.plot(x_data_step, reward_seq, color='red', label="vae_loss")
	# plt.axvline(best_epoch, color='r', linestyle='--', label='best epoch')
	plt.xlabel('Eval_Step')
	plt.ylabel('reward')
	plt.title(f'{plt_label}_{training_iters}')
	plt.legend()
	plt.savefig(os.path.join(fig_path, f'{plt_label}_{training_iters}.png'))
	plt.close()  # 关闭图像，避免出现warning


def policy_visualization(pol_policy, text, fig_path, training_iters):
	from matplotlib import pyplot as plt
	import os

	fig_path = fig_path
	plt_label = text
	policy_seq = pol_policy['policy_seq'][0:100]
	vae_policy_seq = pol_policy['vae_policy_seq'][0:100]
	origin_policy_seq = pol_policy['origin_policy_seq'][0:100]
	state_seq = pol_policy['state_seq'][0:100]
	x_limit_show = 100

	out_c_seq = []
	out_c_setting_seq = []
	for t in range(len(state_seq)):
		out_c_seq.append(state_seq[t][1])
		out_c_setting_seq.append(state_seq[t][-1])

	x_data_step = [x for x in range(len(policy_seq))]
	# best_epoch = y_dataset_eval_rrse.index(min(y_dataset_eval_rrse))

	plt.figure()
	plt.subplot(211)
	plt.plot(x_data_step, out_c_seq, color='red', label="out_c")
	plt.plot(x_data_step, out_c_setting_seq, color='black', label="out_c_setting")
	# plt.xlim(x_limit_show)
	plt.xlabel('train_Step')
	plt.title(f'{plt_label}_{training_iters}')
	plt.ylabel('state')
	plt.legend()

	plt.subplot(212)
	plt.plot(x_data_step, vae_policy_seq, color='red', label="vae_policy")
	plt.plot(x_data_step, policy_seq, color='blue', label="policy")
	plt.plot(x_data_step, origin_policy_seq, color='black', label="origin_policy")
	# plt.axvline(best_epoch, color='r', linestyle='--', label='best epoch')
	# plt.xlim(x_limit_show)
	plt.xlabel('train_Step')
	plt.ylabel('policy')
	plt.legend()

	plt.savefig(os.path.join(fig_path, f'{plt_label}_{training_iters}.png'))
	plt.close()  # 关闭图像，避免出现warning


def set_random_seed(seed):
	rand_seed = np.random.randint(0, 100000) if seed is None else seed
	# logging('random seed = {}'.format(rand_seed))
	print('random seed = {}'.format(rand_seed))
	np.random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed(rand_seed)
