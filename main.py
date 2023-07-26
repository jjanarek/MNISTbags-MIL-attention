import argparse
import time
import timeit
import os
import getpass
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Subset
import torch.optim as optim
from sklearn.model_selection import KFold

from dataset import Bags
from dataset import BalancedBags
from model import CNN
from utils import experiment


parser = argparse.ArgumentParser(description="Ardigen MNIST-bags task model")

# model and training parameters
parser.add_argument("--epochs", type=int, default=200, metavar='N',
                    help="number of epoch to train (default 200)")
parser.add_argument("--model_name", type=str, default='test', metavar='name', help="model name, default 'test'")

parser.add_argument("--self_att", action='store_true', default=False,
                    help='self-attention mode, default False')
parser.add_argument("--kernel_self_att", action='store_true', default=False,
                    help='Gaussian kernel self-attention mode, default False')
parser.add_argument('--L', type=int, default=512, metavar='N',
                    help='parameter for attention (input hidden units, low-dim embedding), default 512')
parser.add_argument('--D', type=int, default=128, metavar='N',
                    help='parameter for attention (internal hidden units), default 128')
parser.add_argument('--K', type=int, default=1, metavar='N',
                    help='parameter for attention (number of attentions), default 1')

parser.add_argument("--init", type=bool, default=True,
                    help="initialize model with Xavier-uniform weights, default True")
parser.add_argument("--no-init", dest='init', action='store_false')

parser.add_argument("--lr", type=float, default=0.00001, metavar='lr',
                    help="learning rate, default 0.00001")
parser.add_argument("--reg", type=float, default=0.00001, metavar='reg',
                    help='L2 regularization constant, default 0.00001')

# training set parameters
parser.add_argument("--target_number", type=int, default=7, metavar="num",
                    help="number used for bag's labelling (target), default 7")
parser.add_argument("--mean_bag_length", type=int, default=10, metavar="N", help="mean bag length (size), default 10")
parser.add_argument("--std_bag_length", type=int, default=2, metavar="N", help="std. dev. of bag length, default 2")
parser.add_argument("--num_bag_train", type=int, default=50, metavar="N",
                    help="number of bags in training set, default 50")
parser.add_argument('--kfold_test', type=int, default=5, metavar='k', help='number of folds, default 5')
parser.add_argument("--patience", type=int, default=5, metavar='N', help='early stopping patience, default 5')

parser.add_argument("--verbose", action='store_true', default=False, help='verbosity, default False')

args = parser.parse_args()

HLINE = "################################################################################################"

if args.kernel_self_att:
    args.kernel_self_att = True
    args.self_att = True


def run(args, seed):
    """
    Main function performing model training and testing using KFold method for
    a given random number generator seed. Creates appropriate dir-tree and saves results.
    """
    args.seed_dir = args.main_dir + "/" + "seed_{}".format(seed)

    if not Path(args.seed_dir).exists():
        os.makedirs(args.seed_dir)

    torch.manual_seed(seed)
    bag_dataset = Bags
    training_dataset = bag_dataset(label_number=args.target_number,
                                   mean_bag_size=args.mean_bag_length,
                                   std_bag_size=args.std_bag_length,
                                   num_of_bags=args.num_bag_train,
                                   seed=seed,
                                   train=True)

    test_set = bag_dataset(label_number=args.target_number,
                           mean_bag_size=args.mean_bag_length,
                           std_bag_size=args.std_bag_length,
                           num_of_bags=1000,
                           seed=seed,
                           train=False)

    folds = KFold(args.kfold_test)

    test_error_list = []
    test_mcc_list = []
    test_auc_list = []

    # Loop over folds
    for i, (train_idxs, val_idxs) in enumerate(folds.split(training_dataset)):
        args.fold_dir = args.seed_dir + "/" + "fold_{}".format(i+1)
        if not Path(args.fold_dir).exists():
            os.makedirs(args.fold_dir)

        if args.verbose:
            print(HLINE)
        print(f"Fold {i+1}/{args.kfold_test}")

        train_set = Subset(training_dataset, train_idxs)
        valid_set = Subset(training_dataset, val_idxs)
        model = CNN(args)

        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.999), weight_decay=args.reg)
        test_error, fold_mmc, fold_auc = experiment(args, model, optimizer, train_set, valid_set, test_set)

        test_error_list.append(test_error)
        test_auc_list.append(fold_auc)
        test_mcc_list.append(fold_mmc)

    seed_output = dict()
    seed_output["test_error_list"] = test_error_list
    seed_output["test_mcc_list"] = test_mcc_list
    seed_output["test_auc_list"] = test_auc_list

    np.savez_compressed(args.seed_dir + "/output.dict", **seed_output)

    mean_test_error = np.mean(test_error_list)
    mean_test_mcc = np.mean(test_mcc_list)
    mean_test_auc = np.mean(test_auc_list)

    return mean_test_error, mean_test_mcc, mean_test_auc


if __name__ == "__main__":
    initial_time = time.asctime()
    hostname = os.uname()[1].split(".")[0]
    print("Python script started on: {}".format(initial_time))
    print("{:>24}: {}".format('from', hostname))
    print("Name of python script: {}".format(os.path.abspath(__file__)))
    print("Script run by: {}\n".format(getpass.getuser()))

    start_time = timeit.default_timer()

    SEEDS = [71, 79, 53, 32, 98]
    device = 'cpu'

    main_dir = args.model_name + "_" + str(args.num_bag_train) + "bags"

    # add relevant info to args before saving
    args.main_dir = main_dir
    args.seeds = SEEDS
    args.device = device

    if not Path(args.main_dir).exists():
        os.makedirs(main_dir)

    with open(Path(args.main_dir) / "run_params.dict", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    test_error_list = []
    test_mcc_list = []
    test_auc_list = []

    # loop over RNG seeds
    for seed in SEEDS:
        print(HLINE+"\n"+HLINE)

        print(f"Processing seed: {seed}\n")
        test_error_seed, test_mcc_seed, test_auc_seed = run(args, seed)
        test_error_list.append(test_error_seed)
        test_mcc_list.append(test_mcc_seed)
        test_auc_list.append(test_auc_seed)

    mean_error = np.mean(test_error_list)
    mean_mcc = np.mean(test_mcc_list)
    mean_auc = np.mean(test_auc_list)

    print(HLINE)
    print(HLINE)
    print("MEAN RESULTS")
    print(f"Calculated over {len(SEEDS)} seeds with {args.kfold_test} folds:")
    print(f"\tTEST ERROR:\t\t{mean_error:.4f}")
    print(f"\tTEST MCC:\t\t{mean_mcc:.4f}")
    print(f"\tTEST AUC:\t\t{mean_auc:.4f}")
    print(HLINE)

    end_time = time.asctime()
    final_time = timeit.default_timer()

    with open(Path(args.main_dir) / "run_results.out", 'w') as f:
        print("Python script started on: {}".format(initial_time), file=f)
        print("{:>24}: {}".format('from', hostname), file=f)
        print("Name of python script: {}".format(os.path.abspath(__file__)), file=f)
        print("Script run by: {}\n".format(getpass.getuser()), file=f)
        print(HLINE, file=f)
        print("MEAN RESULTS", file=f)
        print(f"Calculated over {len(SEEDS)} seeds with {args.kfold_test} folds:", file=f)
        print(f"\tTEST ERROR:\t\t{mean_error:.4f}", file=f)
        print(f"\tTEST MCC:\t\t{mean_mcc:.4f}", file=f)
        print(f"\tTEST AUC:\t\t{mean_auc:.4f}", file=f)
        print(HLINE, file=f)
        print("", file=f)
        print("Python script ended on: {}".format(end_time), file=f)
        print("Total time: {:.2f} seconds".format(final_time - start_time), file=f)

    print()
    print("Python script ended on: {}".format(end_time))
    print("Total time: {:.2f} seconds".format(final_time - start_time))
