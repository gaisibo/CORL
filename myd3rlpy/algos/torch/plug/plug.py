from abc import ABC, abstractmethod


class Plug(ABC):
    @abstractmethod
    def build(self, networks):
        pass
    def change_task(self, task_id):
        pass
