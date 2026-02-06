# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np
import logging
import os
import json
import hashlib
import fcntl
logging.basicConfig(level=logging.INFO)


def print_banner(s, separator="-", num_star=60):
	logging.info(separator * num_star)
	logging.info(s)
	logging.info(separator * num_star)


class Progress:

	def __init__(self, total, name='Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):
		self.total = total
		self.name = name
		self.ncol = ncol
		self.max_length = max_length
		self.indent = indent
		self.line_width = line_width
		self._speed_update_freq = speed_update_freq

		self._step = 0
		self._prev_line = '\033[F'
		self._clear_line = ' ' * self.line_width

		self._pbar_size = self.ncol * self.max_length
		self._complete_pbar = '#' * self._pbar_size
		self._incomplete_pbar = ' ' * self._pbar_size

		self.lines = ['']
		self.fraction = '{} / {}'.format(0, self.total)

		self.resume()

	def update(self, description, n=1):
		self._step += n
		if self._step % self._speed_update_freq == 0:
			self._time0 = time.time()
			self._step0 = self._step
		self.set_description(description)

	def resume(self):
		self._skip_lines = 1
		logging.info('\n')
		self._time0 = time.time()
		self._step0 = self._step

	def pause(self):
		self._clear()
		self._skip_lines = 1

	def set_description(self, params=[]):

		if type(params) == dict:
			params = sorted([
				(key, val)
				for key, val in params.items()
			])

		############
		# Position #
		############
		self._clear()

		###########
		# Percent #
		###########
		percent, fraction = self._format_percent(self._step, self.total)
		self.fraction = fraction

		#########
		# Speed #
		#########
		speed = self._format_speed(self._step)

		##########
		# Params #
		##########
		num_params = len(params)
		nrow = math.ceil(num_params / self.ncol)
		params_split = self._chunk(params, self.ncol)
		params_string, lines = self._format(params_split)
		self.lines = lines

		description = '{} | {}{}'.format(percent, speed, params_string)
		logging.info(description)
		self._skip_lines = nrow + 1

	def append_description(self, descr):
		self.lines.append(descr)

	def _clear(self):
		position = self._prev_line * self._skip_lines
		empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
		logging.info(position)
		logging.info(empty)
		logging.info(position)

	def _format_percent(self, n, total):
		if total:
			percent = n / float(total)

			complete_entries = int(percent * self._pbar_size)
			incomplete_entries = self._pbar_size - complete_entries

			pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
			fraction = '{} / {}'.format(n, total)
			string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent * 100))
		else:
			fraction = '{}'.format(n)
			string = '{} iterations'.format(n)
		return string, fraction

	def _format_speed(self, n):
		num_steps = n - self._step0
		t = time.time() - self._time0
		speed = num_steps / t
		string = '{:.1f} Hz'.format(speed)
		if num_steps > 0:
			self._speed = string
		return string

	def _chunk(self, l, n):
		return [l[i:i + n] for i in range(0, len(l), n)]

	def _format(self, chunks):
		lines = [self._format_chunk(chunk) for chunk in chunks]
		lines.insert(0, '')
		padding = '\n' + ' ' * self.indent
		string = padding.join(lines)
		return string, lines

	def _format_chunk(self, chunk):
		line = ' | '.join([self._format_param(param) for param in chunk])
		return line

	def _format_param(self, param):
		k, v = param
		return '{} : {}'.format(k, v)[:self.max_length]

	def stamp(self):
		if self.lines != ['']:
			params = ' | '.join(self.lines)
			string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
			self._clear()
			logging.info(string)
			self._skip_lines = 1
		else:
			self._clear()
			self._skip_lines = 0

	def close(self):
		self.pause()


class Silent:

	def __init__(self, *args, **kwargs):
		pass

	def __getattr__(self, attr):
		return lambda *args: None


class EarlyStopping(object):
	def __init__(self, tolerance=5, min_delta=0):
		self.tolerance = tolerance
		self.min_delta = min_delta
		self.counter = 0
		self.early_stop = False

	def __call__(self, train_loss, validation_loss):
		if (validation_loss - train_loss) > self.min_delta:
			self.counter += 1
			if self.counter >= self.tolerance:
				return True
		else:
			self.counter = 0
		return False


class ConfigStateManager:
	"""
	分布式超参数搜索状态管理器
	支持多个程序实例同时运行，自动跳过已搜索或正在搜索的配置
	"""
	def __init__(self, state_file=None):
		if state_file is None:
			state_file = os.path.join(os.getcwd(), '.sweep_state.json')
		self.state_file = state_file
		self.lock_file = state_file + '.lock'
		self._ensure_state_file()
	
	def _ensure_state_file(self):
		"""确保状态文件存在"""
		if not os.path.exists(self.state_file):
			with open(self.state_file, 'w') as f:
				json.dump({}, f)
	
	def _get_config_id(self, config):
		"""生成配置的唯一标识符"""
		# 将配置字典排序后转为字符串，然后hash
		config_str = json.dumps(config, sort_keys=True)
		return hashlib.md5(config_str.encode()).hexdigest()
	
	def _load_state(self):
		"""加载状态文件（需要加锁）"""
		try:
			with open(self.state_file, 'r') as f:
				return json.load(f)
		except (json.JSONDecodeError, FileNotFoundError):
			return {}
	
	def _save_state(self, state):
		"""保存状态文件（需要加锁）"""
		with open(self.state_file, 'w') as f:
			json.dump(state, f, indent=2)
	
	def _acquire_lock(self, timeout=10):
		"""获取文件锁"""
		lock_fd = open(self.lock_file, 'w')
		start_time = time.time()
		while True:
			try:
				fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
				return lock_fd
			except IOError:
				if time.time() - start_time > timeout:
					raise TimeoutError(f"Could not acquire lock after {timeout} seconds")
				time.sleep(0.1)
	
	def _release_lock(self, lock_fd):
		"""释放文件锁"""
		try:
			fcntl.flock(lock_fd, fcntl.LOCK_UN)
			lock_fd.close()
		except:
			pass
	
	def get_pending_configs(self, all_configs):
		"""获取所有待处理的配置（pending状态）"""
		lock_fd = None
		try:
			lock_fd = self._acquire_lock()
			state = self._load_state()
			
			pending = []
			for config in all_configs:
				config_id = self._get_config_id(config)
				config_status = state.get(config_id, 'pending')
				if config_status == 'pending':
					pending.append(config)
			
			return pending
		finally:
			if lock_fd:
				self._release_lock(lock_fd)
	
	def try_claim_config(self, config):
		"""
		尝试声明一个配置（从pending转为running）
		返回True如果成功声明，False如果已被其他进程声明
		"""
		config_id = self._get_config_id(config)
		lock_fd = None
		try:
			lock_fd = self._acquire_lock()
			state = self._load_state()
			
			# 检查状态
			current_status = state.get(config_id, 'pending')
			if current_status != 'pending':
				return False  # 已被其他进程处理或正在处理
			
			# 声明为running
			state[config_id] = {
				'status': 'running',
				'config': config,
				'start_time': time.time(),
				'pid': os.getpid()
			}
			self._save_state(state)
			return True
		finally:
			if lock_fd:
				self._release_lock(lock_fd)
	
	def mark_config_completed(self, config):
		"""标记配置为已完成"""
		config_id = self._get_config_id(config)
		lock_fd = None
		try:
			lock_fd = self._acquire_lock()
			state = self._load_state()
			
			state[config_id] = {
				'status': 'completed',
				'config': config,
				'start_time': state.get(config_id, {}).get('start_time', time.time()),
				'end_time': time.time(),
				'pid': os.getpid()
			}
			self._save_state(state)
		finally:
			if lock_fd:
				self._release_lock(lock_fd)
	
	def mark_config_failed(self, config, error_msg=None):
		"""标记配置为失败"""
		config_id = self._get_config_id(config)
		lock_fd = None
		try:
			lock_fd = self._acquire_lock()
			state = self._load_state()
			
			state[config_id] = {
				'status': 'failed',
				'config': config,
				'start_time': state.get(config_id, {}).get('start_time', time.time()),
				'end_time': time.time(),
				'error': str(error_msg) if error_msg else None,
				'pid': os.getpid()
			}
			self._save_state(state)
		finally:
			if lock_fd:
				self._release_lock(lock_fd)
	
	def get_status_summary(self):
		"""获取状态摘要"""
		lock_fd = None
		try:
			lock_fd = self._acquire_lock()
			state = self._load_state()
			
			summary = {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0}
			for config_id, config_data in state.items():
				if isinstance(config_data, dict):
					status = config_data.get('status', 'pending')
				else:
					status = config_data  # 兼容旧格式
				summary[status] = summary.get(status, 0) + 1
			
			return summary
		finally:
			if lock_fd:
				self._release_lock(lock_fd)