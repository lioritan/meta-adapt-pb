# meta-dataset generator, NN model for deterministic, model func for stochastic
import abc


class BaseDatasetLoader(abc.ABC):

    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def get_deterministic_model(self):
        pass

    @abc.abstractmethod
    def get_stochastic_model(self):
        pass

    @abc.abstractmethod
    def train_taskset(self, n_ways, n_shots):
        pass

    @abc.abstractmethod
    def validation_taskset(self, n_ways, n_shots):
        pass

    @abc.abstractmethod
    def test_taskset(self, n_ways, n_shots):
        pass