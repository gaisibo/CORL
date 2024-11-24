from abc import ABC, abstractmethod


class Plug(ABC):
    def __init__(self, algo, networks, update, optim):
        self._algo = algo
        self._networks = networks
        self._update = update
        self._optim = optim
        self.device = algo.device
        self.scaler = algo.scaler
        self.action_scaler = algo.action_scaler
        self.reward_scaler = algo.reward_scaler

    @abstractmethod
    def build(self):
        pass

    def change_task(self, task_id):
        pass

    def pre_loss(self):
        pass

    def pre_task(self):
        pass

    def pre_evaluation(self):
        pass

    def post_loss(self):
        pass

    def post_step(self):
        pass

    def post_task(self):
        pass

    def post_evaluation(self):
        pass
