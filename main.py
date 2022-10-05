import argparse

import wandb

from experiment_runner import ExperimentRunner
from get_by_name import get_dataset_by_name, get_algorithm_by_name


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="mini-imagenet", choices=["mini-imagenet", "omniglot", "mnist"],
                        help="Dataset to use.")
    parser.add_argument('--algorithm', default="train-on-test", choices=["train-on-test", "maml", "bayesian-vi", "meta-adaptation", "vampire"],
                        help="algorithm to use.")
    parser.add_argument('--train_sample_size', default=100, type=int,
                        help="Number of training examples in the inner loop at meta-train time")
    parser.add_argument('--n_ways', default=5, type=int,
                        help="Number of candidate labels (classes) at meta-test time")
    parser.add_argument('--n_shots', default=5, type=int,
                        help="Number of training examples in the inner loop at meta-test time")
    parser.add_argument('--per_task_lr', default=1e-1, type=float,
                        help="Per task LR for adaptation, should be high")
    parser.add_argument('--meta_lr', default=1e-2, type=float,
                        help="Meta LR")
    parser.add_argument('--train_adapt_steps', default=5, type=int,
                        help="Number of gradient steps to take during train adaptation")
    parser.add_argument('--test_adapt_steps', default=10, type=int,
                        help="Number of gradient steps to take during test adaptation")
    parser.add_argument('--meta_batch_size', default=8, type=int,
                        help="Number of task gradients to average for meta-gradient step")
    parser.add_argument('--n_epochs', default=10, type=int,
                        help="Meta epochs for training")
    parser.add_argument('--n_test_epochs', default=40, type=int,
                        help="Meta epochs for test meta-adaptation")
    parser.add_argument('--load_trained_model', default=False, type=bool,
                        help="Load pretrained model")
    parser.add_argument('--test_set_mult', default=5, type=int,
                        help="relative size of evaluation vs adaptation test sets")

    parser.add_argument('--meta_adaptation_is_adaptive', default=True, type=bool,
                        help="KL adapts during run or not")
    parser.add_argument('--mnist_pixels_to_permute_train', default=0, type=int,
                        help="permutes for mnist")
    parser.add_argument('--mnist_pixels_to_permute_test', default=100, type=int,
                        help="permutes for mnist")
    parser.add_argument('--mnist_permute_labels', default=False, type=bool,
                        help="Whether to permute labels")
    parser.add_argument('--seed', type=int, default=7, help="Random seed")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    wandb.init(project="meta-adapt-pb")
    wandb.config.update(args)
    runner = ExperimentRunner(n_ways=args.n_ways, n_shots_train=args.train_sample_size, n_shots_test=args.n_shots,
                              n_epochs_train=args.n_epochs, n_epochs_test=args.n_test_epochs,
                              test_set_mult=args.test_set_mult, load_trained=args.load_trained_model)

    dataset = get_dataset_by_name(args.dataset, args)
    algorithm = get_algorithm_by_name(args.algorithm, args, dataset)

    if not args.load_trained_model:
        runner.run_experiment(algorithm, dataset, seed=args.seed)
        args.load_trained_model = True

    meta_error, meta_accuracy, bound_err, bound_acc = runner.run_experiment(algorithm, dataset, seed=args.seed)
    wandb.log({"test_loss": meta_error, "test_accuracy": meta_accuracy, "bound_loss": bound_err, "bound_acc": bound_acc})
