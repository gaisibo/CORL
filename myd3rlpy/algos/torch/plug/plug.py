from abc import ABC, abstractmethod


class Plug(ABC):
    def __init__(self, algo, networks):
        self._algo = algo
        self._networks = networks
        self.device = algo.device
        self.scaler = algo.scaler
        self.action_scaler = algo.action_scaler
        self.reward_scaler = algo.reward_scaler

    @abstractmethod
    def build(self):
        pass

    def change_task(self, task_id):
        pass
