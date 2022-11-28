import logging

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import factory, loops, network, utils
from inclearn.models import IncrementalLearner

logger = logging.getLogger(__name__)


class FixedRepresentation(IncrementalLearner):

    def __init__(self, args):
        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._lr_decay = args["lr_decay"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]

        logger.info("Initializing FixedRepresentation")

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get(
                "classifier_config", {
                    "type": "fc",
                    "use_bias": True,
                    "use_multi_fc": True
                }
            ),
            device=self._device
        )

        self._n_classes = 0
        self._old_model = None

    def _before_task(self, data_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)

        self._optimizer = factory.get_optimizer(
            self._network.classifier.classifier[-1].parameters(), self._opt_name, self._lr,
            self._weight_decay
        )
        if self._scheduling is None:
            self._scheduler = None
        else:
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self._optimizer, self._scheduling, gamma=self._lr_decay
            )

    def _train_task(self, train_loader, val_loader):
        loops.single_loop(
            train_loader,
            val_loader,
            self._multiple_devices,
            self._network,
            self._n_epochs,
            self._optimizer,
            scheduler=self._scheduler,
            train_function=self._forward_loss,
            eval_function=self._accuracy,
            task=self._task,
            n_tasks=self._n_tasks
        )

    def _after_task(self, inc_dataset):
        self._old_model = self._network.copy().freeze().eval().to(self._device)
        self._network.on_task_end()

    def _eval_task(self, loader):
        ypred, ytrue = [], []

        for input_dict in loader:
            with torch.no_grad():
                logits = self._network(input_dict["inputs"].to(self._device))["logits"]

            ytrue.append(input_dict["targets"].numpy())
            ypred.append(torch.softmax(logits, dim=1).cpu().numpy())

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)

        return ypred, ytrue

    def _accuracy(self, loader):
        ypred, ytrue = self._eval_task(loader)
        ypred = ypred.argmax(dim=1)

        return 100 * round(np.mean(ypred == ytrue), 3)

    def _forward_loss(self, training_network, inputs, targets, memory_flags, metrics):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        outputs = training_network(inputs)

        loss = self._compute_loss(inputs, outputs, targets, onehot_targets, memory_flags, metrics)

        if not utils.check_loss(loss):
            raise ValueError("Loss became invalid ({}).".format(loss))

        metrics["loss"] += loss.item()

        return loss

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags, metrics):
        logits = outputs["logits"]

        return F.cross_entropy(logits, targets)
