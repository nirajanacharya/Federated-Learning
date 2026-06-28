import os
import json
import torch

from src.data.loader       import load_raw
from src.data.eda          import basic_eda
from src.data.preprocessing import split_and_scale
from src.data.partitioner  import partition_noniid
from src.data.builder      import build_client_loaders, make_tensor_ds, make_loader
from src.model.heartnet    import HeartNet
from src.training.client   import FLClient
from src.training.server   import fedavg
from src.evaluation.metrics     import eval_on_loader
from src.visualization.confusion import plot_confusion_matrix
from src.visualization.roc       import plot_roc_curve
from src.visualization.curves    import plot_training_curves, plot_client_losses
from src.utils.logger      import get_logger
from src.utils.seeds       import set_seeds


class FederatedSimulation:
    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger("FedSim", log_file=os.path.join(cfg["log_dir"], "run.log"))
        set_seeds(cfg["seed"])

        self.history = {
            "val_acc": [],
            "val_f1":  [],
            "client_losses": {i: [] for i in range(cfg["num_clients"])}
        }

    def setup_data(self):
        df = load_raw(self.cfg["data_path"])
        basic_eda(df, self.logger)

        X_tr, X_val, X_te, y_tr, y_val, y_te, _ = split_and_scale(df)
        self.input_dim = X_tr.shape[1]

        self.logger.info(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")

        client_indices = partition_noniid(
            X_tr, y_tr,
            num_clients=self.cfg["num_clients"],
            alpha=self.cfg["dirichlet_alpha"],
            seed=self.cfg["seed"]
        )

        for cid, idx in client_indices.items():
            self.logger.debug(f"Client {cid}: {len(idx)} samples")

        self.client_loaders = build_client_loaders(
            X_tr, y_tr, client_indices,
            batch_size=self.cfg["batch_size"]
        )

        self.val_loader  = make_loader(make_tensor_ds(X_val, y_val), batch_size=64)
        self.test_loader = make_loader(make_tensor_ds(X_te,  y_te),  batch_size=64)

    def setup_clients(self):
        self.global_model = HeartNet(self.input_dim).to(self.device)

        self.clients = [
            FLClient(
                client_id=cid,
                dataloader=self.client_loaders[cid],
                model_fn=HeartNet,
                input_dim=self.input_dim,
                lr=self.cfg["lr"],
                local_epochs=self.cfg["local_epochs"],
                device=self.device
            )
            for cid in range(self.cfg["num_clients"])
        ]

        self.logger.info(f"Initialized {len(self.clients)} clients on {self.device}")

    def run(self):
        global_weights = self.global_model.state_dict()

        for rnd in range(1, self.cfg["num_rounds"] + 1):
            client_updates = []

            for c in self.clients:
                c.set_weights(global_weights)
                w, loss, n = c.train_one_round()
                client_updates.append((w, n))
                self.history["client_losses"][c.client_id].append(loss)

            global_weights = fedavg(client_updates)
            self.global_model.load_state_dict(global_weights)

            val_metrics = eval_on_loader(self.global_model, self.val_loader, self.device)
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])

            if rnd % 5 == 0 or rnd == 1:
                self.logger.info(
                    f"Round {rnd:3d} | Val Acc: {val_metrics['accuracy']:.4f} "
                    f"| Val F1: {val_metrics['f1']:.4f} "
                    f"| Val AUC: {val_metrics['roc_auc']:.4f}"
                )

            if rnd % self.cfg["checkpoint_every"] == 0:
                ckpt = os.path.join(self.cfg["checkpoint_dir"], f"global_round_{rnd:03d}.pth")
                torch.save(global_weights, ckpt)

        self.logger.info("Training complete.")
        self._save_history()

    def evaluate_test(self):
        self.logger.info("Running test evaluation...")
        m = eval_on_loader(self.global_model, self.test_loader, self.device)

        self.logger.info(f"Test Accuracy:  {m['accuracy']:.4f}")
        self.logger.info(f"Test Precision: {m['precision']:.4f}")
        self.logger.info(f"Test Recall:    {m['recall']:.4f}")
        self.logger.info(f"Test F1:        {m['f1']:.4f}")
        self.logger.info(f"Test AUC:       {m['roc_auc']:.4f}")

        plot_confusion_matrix(m["confusion_matrix"], self.cfg["fig_dir"], self.logger)
        plot_roc_curve(m["fpr"], m["tpr"], m["roc_auc"], self.cfg["fig_dir"], self.logger)
        plot_training_curves(self.history, self.cfg["fig_dir"], self.logger)
        plot_client_losses(self.history, self.cfg["fig_dir"], self.logger)

        return m

    def _save_history(self):
        out_path = os.path.join(self.cfg["log_dir"], "history.json")
        with open(out_path, "w") as f:
            json.dump({
                "val_acc": self.history["val_acc"],
                "val_f1":  self.history["val_f1"],
                "client_losses": {str(k): v for k, v in self.history["client_losses"].items()}
            }, f, indent=2)
        self.logger.info(f"History saved to {out_path}")
