import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from dataset.gdance_multi_dataset import EdgeGdanceDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton


class STGDance:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=0.0002,
        weight_decay=0.02,
    ):
        # 检查 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        use_baseline_feats = feature_type == "baseline"

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        self.repr_dim = repr_dim = pos_dim + rot_dim + 4
        feature_dim = 35 if use_baseline_feats else 4800

        horizon_seconds = 10  # 5 -> 150  改这里   seq_len
        FPS = 40
        self.horizon = horizon = horizon_seconds * FPS

        # 加载检查点
        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.normalizer = checkpoint["normalizer"]

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,  # horizon   sequence_length  frames=times*hz
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )

        smpl = SMPLSkeleton(device)
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.device = device
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = optim

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def train_loop(self, opt):
        save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
        opt.exp_name = save_dir.split("/")[-1]
        save_dir = Path(save_dir)
        wdir = save_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)

        # 加载数据集
        train_dataset = EdgeGdanceDataset(
            split_file="train_split_sequence_names.txt",
            split="train",
            target_seq_len=400,
            max_persons=3,
        )

        test_dataset = EdgeGdanceDataset(
            split_file="test_split_sequence_names.txt",
            split="test",
            target_seq_len=400,
            max_persons=3,
            normalizer=train_dataset.normalizer,
        )
        self.normalizer = test_dataset.normalizer

        # Load data
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )


        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            self.train()
            for step, (x, cond, filename, wavnames) in enumerate(tqdm(train_data_loader)):
                x = x.to(self.device).float()  # [2, 5, 450, 151]
                cond = cond.to(self.device).float()  # [2, 450, 4800]

                total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(x, cond)
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                avg_loss += loss.detach().cpu().numpy()
                avg_vloss += v_loss.detach().cpu().numpy()
                avg_fkloss += fk_loss.detach().cpu().numpy()
                avg_footloss += foot_loss.detach().cpu().numpy()
                if step % opt.ema_interval == 0:
                    self.diffusion.ema.update_model_average(
                        self.diffusion.master_model, self.diffusion.model
                    )



            # Save model & upload loss to wandb
            if (epoch % opt.save_interval) == 0:
                # log
                avg_loss /= len(train_data_loader)
                avg_vloss /= len(train_data_loader)
                avg_fkloss /= len(train_data_loader)
                avg_footloss /= len(train_data_loader)
                log_dict = {
                    "Train Loss": avg_loss,
                    "V Loss": avg_vloss,
                    "FK Loss": avg_fkloss,
                    "Foot Loss": avg_footloss,
                }
                wandb.log(log_dict)

                self.eval()

                ckpt = {
                    "ema_state_dict": self.diffusion.master_model.state_dict(),
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                    "normalizer": self.normalizer,
                }
                torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))

                print(f"[MODEL SAVED at Epoch {epoch}]")

        wandb.run.finish()
        print("Training complete.")


    def render_sample(
            self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )

    def render_sample_multi(
            self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.device)
        self.diffusion.render_sample_multi(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )
