from dataset_and_model.base_dataset_loader import BaseDatasetLoader
import learn2learn as l2l
import models.stochastic_models

OMNIGLOT_SHAPE = (1, 28, 28)


class OmniglotLoader(BaseDatasetLoader):
    def __init__(self, n_ways, test_mult=2):
        self.n_ways = n_ways
        self.test_mult = test_mult

    def get_deterministic_model(self):
        return l2l.vision.models.OmniglotCNN(self.n_ways)

    def get_name(self):
        return "omniglot"

    def get_stochastic_model(self):
        log_var_init = {"mean": -10, "std": 0.1}
        model = models.stochastic_models.get_model("omniglot", log_var_init, input_shape=OMNIGLOT_SHAPE,
                                                   output_dim=self.n_ways)
        return model

    def get_tasksets(self, n_ways, n_shots):
        return l2l.vision.benchmarks.get_tasksets(
            "omniglot",
            train_samples=n_shots,
            train_ways=n_ways,
            test_samples=self.test_mult * n_shots,
            test_ways=n_ways,
            root='~/data')

    def train_taskset(self, n_ways, n_shots):
        return self.get_tasksets(n_ways, n_shots).train

    def validation_taskset(self, n_ways, n_shots):
        return self.get_tasksets(n_ways, n_shots).validation

    def test_taskset(self, n_ways, n_shots):
        return self.get_tasksets(n_ways, n_shots).test
