#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serverFedCD import FedFCD
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedWASS import FedWASS
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverFedAPEN import FedAPEN
from FLAlgorithms.servers.serverFedDistill import FedDistill
from FLAlgorithms.servers.serverpFedGen import FedGen
from FLAlgorithms.servers.serverFedRCL import FedRCL
from FLAlgorithms.servers.serverFedMut import FedMut
from FLAlgorithms.servers.serverPerFedAvg import PerFedAvg
from FLAlgorithms.servers.serverFedFTG import FedFTG
from utils.model_utils import create_model
from utils.plot_utils import *
import torch


def create_server_n_user(args, i):
    device = torch.device(args.device)
    model = create_model(args.model, args.dataset, args.algorithm, args.numclass).to(device)

    if 'FedFCD' in args.algorithm:
        server = FedFCD(args, model, i)
    elif 'FedWASS' in args.algorithm:
        server = FedWASS(args, model, i)
    elif 'FedAPEN' in args.algorithm:
        server = FedAPEN(args, model, i)
    elif 'FedFTG' in args.algorithm:
        server = FedFTG(args, model, i)
    elif 'PerFedAvg' in args.algorithm:
        server = PerFedAvg(args, model, i)
    elif 'FedGen' in args.algorithm:
        server=FedGen(args, model, i)
    elif 'FedMut' in args.algorithm:
        server=FedMut(args, model, i)
    elif ('FedDistill' in args.algorithm):
        server = FedDistill(args, model, i)
    elif 'FedAvg' in args.algorithm:
        server = FedAvg(args, model, i)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, i)
    elif ('FedRCL' in args.algorithm):
        server = FedRCL(args, model, i)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server

def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args)
        server.test()

def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar-alpha0.1-ratio0.5")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="pFedMe")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to calculate theta approximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=float, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--num_users", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=3, help="running time")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    parser.add_argument("--numclass", type=int, default=10)

    parser.add_argument("--lambda_max", type=float, default=2, help="FedCD Regularization term 1，2,3,4")
    parser.add_argument("--lambda_min", type=float, default=1.0, help="FedCD Regularization term 1")
    parser.add_argument("--ratio", type=float, default=0.5, help="FedCD training_ratio")
    parser.add_argument("--min_sample", type=int, default=3, help="FedCD min_sample(DBSCAN)")


    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learning rate: {}".format(args.learning_rate))
    print("Ensemble learning rate: {}".format(args.ensemble_lr))
    print("Average Moving: {}".format(args.beta))
    print("Subset of users: {}".format(args.num_users))
    print("Number of global rounds: {}".format(args.num_glob_iters))
    print("Number of local rounds: {}".format(args.local_epochs))
    print("Dataset: {}".format(args.dataset))
    print("Local Model: {}".format(args.model))
    print("Device: {}".format(args.device))
    print("=" * 80)
    main(args)