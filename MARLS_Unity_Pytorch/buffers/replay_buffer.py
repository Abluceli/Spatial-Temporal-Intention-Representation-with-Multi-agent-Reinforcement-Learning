import numpy as np
import random

random.seed = 3


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, image, state, action, reward, image_, state_, done):
        data = (image, state, action, reward, image_, state_, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        image, state, action, reward, image_, state_, done = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            image_t, state_t, action_t, reward_t, image__t, state__t, done_t = data
            image.append(image_t)
            state.append(state_t)
            action.append(action_t)
            reward.append(reward_t)
            image_.append(image__t)
            state_.append(state__t)
            done.append(done_t)
        return np.array(image, dtype=np.float32), \
            np.array(state, dtype=np.float32), \
            np.array(action, dtype=np.float32), \
            np.array(reward, dtype=np.float32), \
            np.array(image_, dtype=np.float32), \
            np.array(state_, dtype=np.float32), \
            np.array(done, dtype=np.float32)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class ReplayBuffer_Intention_DQN(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, image, intention, state, action, reward, image_, intention_, state_, done):
        data = (image, intention, state, action, reward, image_, intention_, state_, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        image, intention, state, action, reward, image_, intention_, state_, done = [], [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            image_t, intention_t, state_t, action_t, reward_t, image__t, intention__t, state__t, done_t = data
            image.append(image_t)
            intention.append(intention_t)
            state.append(state_t)
            action.append(action_t)
            reward.append(reward_t)
            image_.append(image__t)
            intention_.append(intention__t)
            state_.append(state__t)
            done.append(done_t)
        return np.array(image, dtype=np.float32), \
            np.array(intention, dtype=np.float32), \
            np.array(state, dtype=np.float32), \
            np.array(action, dtype=np.float32), \
            np.array(reward, dtype=np.float32), \
            np.array(image_, dtype=np.float32), \
            np.array(intention_, dtype=np.float32), \
            np.array(state_, dtype=np.float32), \
            np.array(done, dtype=np.float32)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class ReplayBuffer_QMIX(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, state, reward, obs_tp1, done, state_next):
        data = (obs_t, action, state, reward, obs_tp1, done, state_next)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, states, rewards, obses_tp1, dones, states_next = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, state, reward, obs_tp1, done, state_next = data
            obses_t.append(obs_t)
            actions.append(action)
            states.append(state)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
            states_next.append(state_next)
        return np.array(obses_t, dtype=np.float32), \
            np.array(actions, dtype=np.float32), \
            np.array(states, dtype=np.float32), \
            np.array(rewards, dtype=np.float32), \
            np.array(obses_tp1, dtype=np.float32), \
            np.array(dones, dtype=np.float32), \
            np.array(states_next, dtype=np.float32)

        # return obses_t, actions, rewards, obses_tp1, dones

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
