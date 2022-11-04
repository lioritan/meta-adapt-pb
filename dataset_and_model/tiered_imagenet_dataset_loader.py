from dataset_and_model.base_dataset_loader import BaseDatasetLoader
import learn2learn as l2l
import models.stochastic_models
import functools

INET_SHAPE = (3, 84, 84)


class TieredImagenetLoader(BaseDatasetLoader):
    def __init__(self, n_ways, test_mult=2):
        self.n_ways = n_ways
        self.test_mult = test_mult

    def get_deterministic_model(self):
        # usually methods use resnet for this
        # return l2l.vision.models.ResNet12(self.n_ways)
        #return l2l.vision.models.OmniglotFC(functools.reduce(lambda x, y: x * y, INET_SHAPE), self.n_ways,
        #                                    sizes=[128, 32])
        return l2l.vision.models.MiniImagenetCNN(self.n_ways)

    def get_stochastic_model(self):
        # TODO
        log_var_init = {"mean": -10, "std": 0.1}
        model = models.stochastic_models.get_model("mini-imagenet", log_var_init, input_shape=INET_SHAPE,
                                                   output_dim=self.n_ways)
        return model

    def get_name(self):
        return "tiered-imagenet"

    def get_tasksets(self, n_ways, n_shots):
        # Note: the download function is currently broken due to a change in google drive
        # File can be downloaded manually at https://docs.google.com/uc?export=download&id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07
        return l2l.vision.benchmarks.get_tasksets(
            "tiered-imagenet",
            train_samples=n_shots,
            train_ways=n_ways,
            test_samples=self.test_mult * n_shots,
            test_ways=n_ways,
            root='./data', download=False)

    def train_taskset(self, n_ways, n_shots):
        return self.get_tasksets(n_ways, n_shots).train

    def validation_taskset(self, n_ways, n_shots):
        return self.get_tasksets(n_ways, n_shots).validation

    def test_taskset(self, n_ways, n_shots):
        return self.get_tasksets(n_ways, n_shots).test
