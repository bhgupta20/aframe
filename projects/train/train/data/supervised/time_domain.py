import torch

from ml4gw.transforms.decimator import Decimator
from train.data.supervised.supervised import SupervisedAframeDataset


class TimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.hparams.schedule is not None:
            schedule = self.hparams.schedule
            self.schedule = torch.tensor(schedule, dtype=torch.int)
            self.decimator = Decimator(sample_rate=self.hparams.sample_rate, schedule=self.schedule)

    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)

        if self.decimator is not None:
            X_bg = self.decimator(X_bg)
            X_fg = self.decimator(X_fg)
        return X_bg, X_fg

    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        if self.decimator is not None:
            X = self.decimator(X)
        return X, y
    
class TimeDomainSupervisedDecimateAframeDataset(SupervisedAframeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.hparams.schedule is not None:
            schedule = self.hparams.schedule
            self.schedule = torch.tensor(schedule, dtype=torch.int)
            self.decimator = Decimator(sample_rate=self.hparams.sample_rate, schedule=self.schedule)

    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)

        if self.decimator is not None:
            X_bg = self.decimator(X_bg)
            X_fg = self.decimator(X_fg)
        return X_bg, X_fg
    
    def on_before_batch_transfer(self, batch, _):
        """
        Slice loaded waveforms before sending to device
        """
        # TODO: maybe pass indices as argument to
        # waveform loader to reduce quantity of data
        # we need to load
        if self.trainer.training:
            X, waveforms = batch
            waveforms = self.slice_waveforms(waveforms)
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            batch = self.augment(X, waveforms)
        elif self.trainer.validating or self.trainer.sanity_checking:
            # If we're in validation mode but we're not validating
            # on the local device, the relevant tensors will be
            # empty, so just pass them through with a 0 shift to
            # indicate that this should be ignored
            [background, _, timeslide_idx], [signals] = batch

            # If we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU. Once everything is
            # on-device, pre-inject signals into background.
            shift = self.timeslides[timeslide_idx].shift_size
            X_bg, X_fg = self.build_val_batches(background, signals)
            batch = (shift, X_bg, X_fg)
        return batch

    def on_after_batch_transfer(self, batch, _):
        """
        This is a method inherited from the DataModule
        base class that gets called after data returned
        by a dataloader gets put on the local device,
        but before it gets passed to the LightningModule.
        Use this to do on-device augmentation/preprocessing.
        """
        return batch

    def inject(self, X, waveforms):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        if self.decimator is not None:
            X = self.decimator(X)
        return X, y
    