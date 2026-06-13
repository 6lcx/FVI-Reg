import os
import time
from collections import OrderedDict

import numpy as np
import torch
from Metrics import MetricTest
from Modules.Loss import DiceCoefficientAll
from Networks import BaseRegistraionNetwork
from rich.progress import BarColumn, Progress, TimeRemainingColumn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils import EarlyStopping


progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(),
    "[progress.percentage]{task.percentage:3.2f}%",
    "{task.completed:5.0f}",
    "best: {task.fields[best]:.5f}",
    "best_epoch: {task.fields[best_epoch]:5.0f}",
    TimeRemainingColumn(),
)


class BaseController_Seg_GSFVI:
    def __init__(self, net: BaseRegistraionNetwork):
        self.net = net

    def cuda(self):
        self.net.cuda()

    @staticmethod
    def _checkpoint_state(payload):
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        elif isinstance(payload, dict) and "model" in payload:
            payload = payload["model"]
        if any(key.startswith("module.") for key in payload):
            payload = OrderedDict(
                (key.removeprefix("module."), value) for key, value in payload.items()
            )
        return payload

    @staticmethod
    def _prefixed_state(state, prefix):
        return OrderedDict(
            (key[len(prefix):], value)
            for key, value in state.items()
            if key.startswith(prefix)
        )

    @classmethod
    def _segmentation_state(cls, checkpoint_path):
        state = cls._checkpoint_state(torch.load(checkpoint_path, map_location="cpu"))
        segmented_state = cls._prefixed_state(state, "seg_F.")
        return segmented_state or state

    def _load_frozen_priors(self):
        reg_checkpoint = "/root/dym/train_result/SegFormer_Det/mm-LCC---4.0-30000-20260504004014/best.pt"
        if not os.path.exists(reg_checkpoint):
            reg_checkpoint = os.path.join(
                os.getcwd(),
                "train_result",
                "SegFormerRBF_Det",
                "mm-LCC---4.0-30000-20260504004014",
                "best.pt",
            )

        seg_checkpoint = "/root/dym/train_result/REG_RBFGI/MM-LCC---[9, 9]--147000-20251229005847/best.pt"
        if not os.path.exists(seg_checkpoint):
            seg_checkpoint = os.path.join(
                os.getcwd(),
                "train_result",
                "REG_RBFGI",
                "MM-LCC---[9, 9]--147000-20251229005847",
                "best.pt",
            )

        reg_state = self._checkpoint_state(torch.load(reg_checkpoint, map_location="cpu"))
        self.net.reg_pretrain.load_state_dict(
            self._prefixed_state(reg_state, "encoder."), strict=False
        )
        seg_state = (
            self._segmentation_state(seg_checkpoint)
            if os.path.exists(seg_checkpoint)
            else self._prefixed_state(reg_state, "seg_F.")
        )
        self.net.seg_F.load_state_dict(seg_state, strict=False)

        for module in (self.net.reg_pretrain, self.net.seg_F):
            module.requires_grad_(False)
            module.eval()

    def train(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        save_checkpoint,
        earlystop: EarlyStopping,
        logger: SummaryWriter,
        start_epoch=0,
        max_epoch=100,
        lr=3e-4,
        v_step=50,
        verbose=1,
        **_,
    ):
        self._load_frozen_priors()
        optimizer = torch.optim.Adam(
            (parameter for parameter in self.net.parameters() if parameter.requires_grad),
            lr=lr,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=1.0
        )
        earlystop.on_train_begin()

        task = None
        if verbose == 0:
            task = progress.add_task(
                "Training...", total=max_epoch, best=0, best_epoch=0
            )
            progress.start()

        end = time.perf_counter()
        for epoch in range(start_epoch, max_epoch):
            start = end
            self.net.train()
            self.net.reg_pretrain.eval()
            self.net.seg_F.eval()
            train_loss_dict = self.trainIter(train_dataloader, optimizer)
            scheduler.step()

            self.net.eval()
            validation_dice = self.validationIter(validation_dataloader)

            if save_checkpoint:
                save_checkpoint(self.net, epoch + 1)

            end = time.perf_counter()
            if verbose:
                losses = ", ".join(
                    f"{key}: {value:.6f}" for key, value in train_loss_dict.items()
                )
                print(epoch + 1, f"{end - start:.2f}", losses, validation_dice)

            should_stop = earlystop.on_epoch_end(epoch + 1, validation_dice, self.net)
            if task is not None:
                progress.update(
                    task,
                    advance=1,
                    best=earlystop.best,
                    best_epoch=earlystop.best_epoch,
                    refresh=True,
                )
            if should_stop:
                break

        if task is not None:
            progress.stop_task(task)
            progress.remove_task(task)
            progress.stop()
        return earlystop.best

    def trainIter(
        self, dataloader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> dict:
        losses = {}
        for data in dataloader:
            src = data["src"]["img"].cuda()
            tgt = data["tgt"]["img"].cuda()

            optimizer.zero_grad()
            loss_dict = self.net.objective(src, tgt)
            loss_dict["loss"].mean().backward()
            optimizer.step()

            for key, value in loss_dict.items():
                losses.setdefault(key, []).append(value.mean().item())

        return {key: np.mean(values) for key, values in losses.items()}

    def validationIter(self, dataloader: DataLoader):
        dice_list = []
        dice_estimator = DiceCoefficientAll()
        with torch.no_grad():
            for data in dataloader:
                src = data["src"][0].cuda().float()
                tgt = data["tgt"][0].cuda().float()
                src_seg = data["src_seg"][0].cuda().float()
                tgt_seg = data["tgt_seg"][0].cuda().int()

                phi = self.net.test(src, tgt)[0]
                warped_src_seg = self.net.transformer(src_seg, phi, mode="nearest")
                dice_list.append(
                    dice_estimator(tgt_seg, warped_src_seg.int()).unsqueeze(0)
                )

        return torch.cat(dice_list, 0).mean().item()

    def test(
        self,
        dataloader: DataLoader,
        name: str = None,
        network: str = None,
        excel_save_path: str = None,
        verbose=2,
    ):
        self.net.eval()
        metric_test = MetricTest()
        with torch.no_grad():
            for data in dataloader:
                src = data["src"][0].cuda().float()
                tgt = data["tgt"][0].cuda().float()
                src_seg = data["src_seg"][0].cuda().float()
                tgt_seg = data["tgt_seg"][0].cuda().float()
                case_no = data["case_no"].item()
                slice_index = data["slice"]
                resolution = data["resolution"].item()

                phi_src_to_tgt = self.net.test(src, tgt)[0]
                phi_tgt_to_src = self.net.test(tgt, src)[0]
                warped_src_seg = self.net.transformer(
                    src_seg, phi_src_to_tgt, mode="nearest"
                )
                warped_tgt_seg = self.net.transformer(
                    tgt_seg, phi_tgt_to_src, mode="nearest"
                )

                metric_test.testMetrics(
                    src_seg.int(),
                    warped_src_seg.int(),
                    tgt_seg.int(),
                    warped_tgt_seg.int(),
                    resolution,
                    case_no,
                    slice_index,
                )
                metric_test.testFlow(phi_src_to_tgt, phi_tgt_to_src, case_no)

        mean = metric_test.mean()
        if verbose >= 2:
            metric_test.saveAsExcel(
                network, name, os.path.join(excel_save_path, network)
            )
        return mean, metric_test.details
