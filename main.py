import argparse

import wandb

from experiment_runner import ExperimentRunner
from get_by_name import get_dataset_by_name, get_algorithm_by_name


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="omniglot", choices=["mini-imagenet", "omniglot", "mnist", "tiered-imagenet"],
                        help="Dataset to use.")
    parser.add_argument('--algorithm', default="vampire", choices=["train-on-test", "maml", "bayesian-vi", "meta-adaptation", "vampire", "bmaml"],
                        help="algorithm to use.")
    parser.add_argument('--train_sample_size', default=5, type=int,
                        help="Number of training examples in the inner loop at meta-train time")
    parser.add_argument('--n_ways', default=5, type=int,
                        help="Number of candidate labels (classes) at meta-test time")
    parser.add_argument('--n_shots', default=1, type=int,
                        help="Number of training examples in the inner loop at meta-test time")
    parser.add_argument('--per_task_lr', default=1e-1, type=float,
                        help="Per task LR for adaptation, should be high")
    parser.add_argument('--meta_lr', default=1e-3, type=float,
                        help="Meta LR")
    parser.add_argument('--train_adapt_steps', default=5, type=int,
                        help="Number of gradient steps to take during train adaptation")
    parser.add_argument('--test_adapt_steps', default=10, type=int,
                        help="Number of gradient steps to take during test adaptation")
    parser.add_argument('--meta_batch_size', default=4, type=int,
                        help="Number of task gradients to average for meta-gradient step")
    parser.add_argument('--n_epochs', default=500, type=int,
                        help="Meta epochs for training")
    parser.add_argument('--n_test_epochs', default=1, type=int,
                        help="Meta epochs for test meta-adaptation")
    # Note: boolean values don't work in argparse, so if this is set it is always true
    parser.add_argument('--load_trained_model', default=False, type=bool,
                        help="Load pretrained model")
    parser.add_argument('--test_set_mult', default=2, type=int,
                        help="relative size of evaluation vs adaptation test sets")

    # Note: boolean values don't work in argparse, so if this is set it is always true
    parser.add_argument('--meta_adaptation_is_adaptive', default=False, type=bool,
                        help="KL adapts during run or not")
    parser.add_argument('--vi_hyper_kl_test_factor', type=float, default=0,
                        help="Multiplicative weight of hyper-kl in meta-testing for vi/meta-adaptation")
    parser.add_argument('--vampire_kl_weight', default=1e-6, type=float,
                        help="Relative KL weight")
    parser.add_argument('--vampire_num_models', default=3, type=int,
                        help="number of models/MC averages for vampire")
    parser.add_argument('--vampire_num_models_test', default=3, type=int,
                        help="number of models/MC averages for vampire test")
    parser.add_argument('--bmaml_num_particles', default=3, type=int,
                        help="number of particles for Bmaml (SVGD models)")
    parser.add_argument('--mnist_pixels_to_permute_train', default=0, type=int,
                        help="permutes for mnist")
    parser.add_argument('--mnist_pixels_to_permute_test', default=100, type=int,
                        help="permutes for mnist")
    # Note: boolean values don't work in argparse, so if this is set it is always true
    parser.add_argument('--mnist_permute_labels', default=False, type=bool,
                        help="Whether to permute labels")
    parser.add_argument('--optimizer_weight_decay', default=0, type=float,
                        help="Weight decay parameter for optimizer")
    parser.add_argument('--optimizer_lr_decay_epochs', default=10, type=int,
                        help="Number of epochs until lr decay for step schedule")
    parser.add_argument('--optimizer_lr_schedule_type', default="no_change", choices=["no_change", "step"],
                        help="Type of lr schedule")
    # Note: boolean values don't work in argparse, so if this is set it is always true
    parser.add_argument('--early_stop', default=True, type=bool, help="early stop on validation")

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

    meta_error, meta_accuracy, bound_err, bound_acc = runner.run_experiment(algorithm, dataset, seed=args.seed)
    wandb.log({"test_loss": meta_error, "test_accuracy": meta_accuracy, "bound_loss": bound_err, "bound_acc": bound_acc})
