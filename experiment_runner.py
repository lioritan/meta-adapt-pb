from dataset_and_model.base_dataset_loader import BaseDatasetLoader
from meta_learning.base_meta_learner import BaseMetaLearner
from utils.common import set_random_seed
import os


class ExperimentRunner(object):
    def __init__(self, n_ways, n_shots_train, n_shots_test, n_epochs_train, n_epochs_test, test_set_mult, load_trained):
        self.load_trained = load_trained
        self.test_set_mult = test_set_mult
        self.n_epochs_test = n_epochs_test
        self.n_epochs_train = n_epochs_train
        self.n_shots_test = n_shots_test
        self.n_shots_train = n_shots_train
        self.n_ways = n_ways
        self.algorithms = []
        self.datasets = []

    def add_algorithm(self, learner: BaseMetaLearner):
        self.algorithms.append(learner)

    def add_dataset(self, dataset: BaseDatasetLoader):
        self.datasets.append(dataset)

    def run_experiment_suite(self, seed=1):
        for algorithm in self.algorithms:
            for dataset in self.datasets:
                self.run_experiment(algorithm, dataset, seed)
                #TODO: report

    def run_experiment(self, algorithm: BaseMetaLearner, dataset: BaseDatasetLoader, seed):
        model_name = f"artifacts/{dataset.get_name()}/{str(algorithm.__class__.__name__)}.pkl"
        os.makedirs(f"artifacts/{dataset.get_name()}", exist_ok=True)
        print(self.load_trained)
        if self.load_trained:
            print(f"load trained model")
            algorithm.load_saved_model(model_name=model_name)
        else:
            print(f"meta learner train")
            set_random_seed(seed)
            algorithm.meta_train(dataset.train_taskset(self.n_ways, self.n_shots_train),
                                 dataset.validation_taskset(self.n_ways, self.n_shots_train),
                                 self.n_epochs_train)
            algorithm.save_model(model_name=model_name)

        set_random_seed(seed)
        return algorithm.meta_test(dataset.test_taskset(self.n_ways, self.n_shots_test),
                                   self.n_epochs_test, self.test_set_mult)
