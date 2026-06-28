import os
import argparse

from src.training.simulation import FederatedSimulation


def parse_args():
    p = argparse.ArgumentParser(description="Federated Learning — Heart Disease Dataset")
    p.add_argument("--data_path",        type=str,   default="data/heart.csv")
    p.add_argument("--num_clients",      type=int,   default=5)
    p.add_argument("--num_rounds",       type=int,   default=30)
    p.add_argument("--local_epochs",     type=int,   default=5)
    p.add_argument("--lr",               type=float, default=0.001)
    p.add_argument("--batch_size",       type=int,   default=16)
    p.add_argument("--dirichlet_alpha",  type=float, default=0.5)
    p.add_argument("--checkpoint_every", type=int,   default=10)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--fig_dir",          type=str,   default="outputs/figures")
    p.add_argument("--checkpoint_dir",   type=str,   default="outputs/checkpoints")
    p.add_argument("--log_dir",          type=str,   default="outputs/logs")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = vars(args)

    for d in [cfg["fig_dir"], cfg["checkpoint_dir"], cfg["log_dir"]]:
        os.makedirs(d, exist_ok=True)

    sim = FederatedSimulation(cfg)
    sim.setup_data()
    sim.setup_clients()
    sim.run()
    sim.evaluate_test()


if __name__ == "__main__":
    main()
