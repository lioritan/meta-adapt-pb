import argparse
import wandb
import numpy as np

from meta_learner_run_pb import run_meta_learner


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="mnist", choices=["mini-imagenet", "omniglot", "mnist"],
                        help="Dataset to use.")
    parser.add_argument('--train_sample_size', default=100, type=int,
                        help="Number of training examples in the inner loop at meta-train time")
    parser.add_argument('--n_ways', default=5, type=int,
                        help="Number of candidate labels (classes) at meta-test time")
    parser.add_argument('--n_shots', default=50, type=int,
                        help="Number of training examples in the inner loop at meta-test time")
    parser.add_argument('--per_task_lr', default=1e-1, type=float,
                        help="Per task LR for adaptation, should be high")
    parser.add_argument('--meta_lr', default=1e-2, type=float,
                        help="Meta LR")
    parser.add_argument('--train_adapt_steps', default=50, type=int,
                        help="Number of gradient steps to take during train adaptation")
    parser.add_argument('--test_adapt_steps', default=50, type=int,
                        help="Number of gradient steps to take during test adaptation")
    parser.add_argument('--meta_batch_size', default=8, type=int,
                        help="Number of task gradients to average for meta-gradient step")
    parser.add_argument('--n_epochs', default=100, type=int,
                        help="Meta epochs for training")
    parser.add_argument('--reset_clf_on_meta', default=False, type=bool,
                        help="Should the clf layer be reset each meta loop (should make adaptation faster)")
    parser.add_argument('--n_test_epochs', default=40, type=int,
                        help="Meta epochs for test meta-adaptation")
    parser.add_argument('--load_trained_model', default=True, type=bool,
                        help="Load pretrained model")
    parser.add_argument('--is_adaptive', default=True, type=bool,
                        help="KL adapts during run or not")
    parser.add_argument('--mnist_pixels_to_permute_train', default=0, type=int,
                        help="permutes for mnist")
    parser.add_argument('--mnist_pixels_to_permute_test', default=100, type=int,
                        help="permutes for mnist")
    parser.add_argument('--mnist_permute_labels', default=False, type=bool,
                        help="Whether to permute labels")
    parser.add_argument('--seed', type=int, default=7, help="Random seed")
    return parser


def run_experiment(args):
    experiment_result = run_meta_learner(
        dataset=args.dataset,
        train_sample_size=args.train_sample_size,
        n_ways=args.n_ways,
        n_shots=args.n_shots,
        per_task_lr=args.per_task_lr,
        meta_lr=args.meta_lr,
        train_adapt_steps=args.train_adapt_steps,
        test_adapt_steps=args.test_adapt_steps,
        meta_batch_size=args.meta_batch_size,
        n_epochs=args.n_epochs,
        reset_clf_on_meta_loop=args.reset_clf_on_meta,
        n_test_epochs=args.n_test_epochs,
        load_trained=args.load_trained_model,
        is_adaptive=args.is_adaptive,
        mnist_pixels_to_permute_train=args.mnist_pixels_to_permute_train,
        mnist_pixels_to_permute_test=args.mnist_pixels_to_permute_test,
        mnist_permute_labels=args.mnist_permute_labels,
        seed=args.seed)
    meta_error, meta_accuracy, bound_err, bound_acc = \
        experiment_result[0], experiment_result[1], experiment_result[2], experiment_result[3]
    return meta_error, meta_accuracy, bound_err, bound_acc


if __name__ == "__main__":
    args = get_parser().parse_args()
    wandb.init(project="meta-pb-stochastic")
    wandb.config.update(args)

    if not args.load_trained_model:
        run_experiment(args)
        args.load_trained_model = True

    meta_error, meta_accuracy, bound_err, bound_acc = run_experiment(args)
    wandb.log({"test_loss": meta_error, "test_accuracy": meta_accuracy, "bound_loss": bound_err, "bound_acc": bound_acc})
