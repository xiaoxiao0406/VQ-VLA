import numpy as np


class TemporalAgg:
    def __init__(
        self,
        apply=False,
        action_dim=8,
        chunk_size=20,
        k=0.01,
    ) -> None:
        self.apply = apply

        if self.apply:
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.action_buffer = np.zeros((self.chunk_size, self.chunk_size, self.action_dim))
            self.full_action = False
            self.k = k

    def reset(self):
        self.action_buffer = np.zeros((self.chunk_size, self.chunk_size, self.action_dim))

    def add_action(self, action):
        if not self.full_action:
            t = ((self.action_buffer != 0).sum(1).sum(1) != 0).sum()
            self.action_buffer[t] = action
            if t == self.chunk_size - 1:
                self.full_action = True
        else:
            self.action_buffer = np.roll(self.action_buffer, -1, axis=0)
            self.action_buffer[-1] = action

    def get_action(self):
        actions_populated = (
            ((self.action_buffer != 0).sum(1).sum(1) != 0).sum() if not self.full_action else self.chunk_size
        )
        exp_weights = np.exp(-np.arange(actions_populated) * self.k)
        exp_weights = exp_weights / exp_weights.sum()
        current_t_actions = self.action_buffer[:actions_populated][
            np.eye(self.chunk_size)[::-1][-actions_populated:].astype(bool)
        ]
        return (current_t_actions * exp_weights[:, None]).sum(0)

    def __call__(self, action):
        if not self.apply:
            return action[0]
        else:
            self.add_action(action)
            return self.get_action()
