import copy
import collections
import logging
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from inclearn.lib import factory, herding, losses, network, utils
from inclearn.models.icarl import ICarl

logger = logging.getLogger(__name__)


class DER(ICarl):
    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._n_pretrain_epochs = args.get("pretrain_epochs", self._n_epochs)

        self._scheduling = args["scheduling"]
        self._pretrain_scheduling = args.get("pretrain_scheduling")
        self._lr_decay = args["lr_decay"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._n_classes = 0

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})
        
        self._dist_loss = args.get("distillation_loss")
        self._aux_loss = args.get("auxillary_loss")
        self._ranking_loss = args.get("ranking_loss")
        self._class_loss = args.get("classification_loss")

        self._network = network.DynamicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {}),
            aux_classifier_kwargs=args.get("aux_classifier_config", {}),
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=True,
        )

        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._data_memory, self._targets_memory = None, None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")
        self._warmup_config = args.get("warmup_config")
        
        self._reset_classifier = args.get("reset_classifier", False)

        self._weight_generation = args.get("weight_generation")

        self._herding_indexes = []

        self._eval_type = args.get("eval_type", "nme")

        self._meta_transfer = args.get("meta_transfer", False)
        if self._meta_transfer:
            assert args["convnet"] == "rebuffi_mtl"

        self._args = args
        self._args["_logs"] = {}

        self._during_finetune = False
        self._clip_classifier = None
        self._align_weights_after_epoch = False

    def _after_task(self, inc_dataset):
        if "scale" not in self._args["_logs"]:
            self._args["_logs"]["scale"] = []

        if self._network.post_processor is None:
            s = None
        elif hasattr(self._network.post_processor, "scale"):
            s = self._network.post_processor.scale.item()
        elif hasattr(self._network.post_processor, "factor"):
            s = self._network.post_processor.factor.item()

        print("Scale is {}.".format(s))
        self._args["_logs"]["scale"].append(s)

        del self._class_loss['alpha']
        del self._class_loss['alpha_old']
        del self._class_loss['alpha_pretrain']
        del self._aux_loss['alpha']
            
        super()._after_task(inc_dataset)

    def _eval_task(self, data_loader):
        if self._eval_type == "nme":
            return super()._eval_task(data_loader)
        elif self._eval_type == "cnn":
            ypred = []
            ytrue = []

            for input_dict in data_loader:
                ytrue.append(input_dict["targets"].numpy())

                inputs = input_dict["inputs"].to(self._device)
                logits = self._network(inputs)["logits"].detach()

                preds = F.softmax(logits, dim=-1)
                ypred.append(preds.cpu().numpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            self._last_results = (ypred, ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._eval_type)

    def _gen_weights(self):
        # Initialize new parameters at each incremental step
        if self._weight_generation:
            utils.add_new_weights(
                self._network, self._weight_generation if self._task != 0 else "basic",
                self._n_classes, self._task_size, self.inc_dataset
            )

    def _before_task(self, train_loader, val_loader):
        # Expand architecture (Algorithm 1 Line 3)
        self._gen_weights()

        self._n_old_classes = self._n_classes
        self._n_classes += self._task_size
        print("Now {} examplars per class.".format(self._memory_per_class))

        samples_per_cls = np.zeros(self._n_classes)
        all_targets = self.inc_dataset.targets_train
        for i in range(self._n_old_classes, self._n_classes):
            samples_per_cls[i] = np.sum(all_targets==i)
        samples_per_cls[:self._n_old_classes] = self._memory_per_class
        print("Now {} training samples per class.".format(samples_per_cls))
        
        # Set parameters of class-balanced focal loss for classification loss in L^new
        if self._class_loss['beta']:
            effective_num = 1.0 - np.power(self._class_loss['beta'], samples_per_cls)
            weights = (1.0 - self._class_loss['beta']) / np.array(effective_num)
            weights = weights / np.sum(weights) * len(samples_per_cls)
            self._class_loss['alpha'] = torch.tensor(weights).float().to(self._device)     
        else:
            self._class_loss['alpha'] = None

        # Set parameters of class-balanced focal loss for classification loss in first task (Algorithm 1 Line 1)
        if self._class_loss.get('beta_pretrain', 0) > 0:
            effective_num = 1.0 - np.power(self._class_loss['beta_pretrain'], samples_per_cls)
            weights = (1.0 - self._class_loss['beta_pretrain']) / np.array(effective_num)
            weights = weights / np.sum(weights) * len(samples_per_cls)
            self._class_loss['alpha_pretrain'] = torch.tensor(weights).float().to(self._device)     
        else:
            self._class_loss['alpha_pretrain'] = None
        
        # Set parameters of class-balanced focal loss for classification loss in L^old
        if self._class_loss['beta_old'] and self._class_loss['schedule_old'] > 0:
            effective_num = 1.0 - np.power(self._class_loss['beta_old'], samples_per_cls)
            weights = (1.0 - self._class_loss['beta_old']) / np.array(effective_num)
            weights = weights / np.sum(weights) * len(samples_per_cls)
            self._class_loss['alpha_old'] = torch.tensor(weights).float().to(self._device)     
        else:
            self._class_loss['alpha_old'] = None
            
        # Set parameters of class-balanced focal loss for auxiliary loss
        if self._task > 0 and self._aux_loss['factor'] > 0 and self._aux_loss['beta']:
            if self._aux_loss['n+1']:
                samples_per_cls_aux = np.zeros(self._task_size+1)
                samples_per_cls_aux[1:] = samples_per_cls[self._n_old_classes:]
                samples_per_cls_aux[0] = np.sum(samples_per_cls[:self._n_old_classes])
                effective_num = 1.0 - np.power(self._aux_loss['beta'], samples_per_cls_aux)
            else:                
                effective_num = 1.0 - np.power(self._aux_loss['beta'], samples_per_cls)
            weights = (1.0 - self._aux_loss['beta']) / np.array(effective_num)
            weights = weights / np.sum(weights) * len(samples_per_cls)
            self._aux_loss['alpha'] = torch.tensor(weights).float().to(self._device)     
        else:
            self._aux_loss['alpha'] = None

        # Set learning rate for different groups of parameters
        if self._groupwise_factors and isinstance(self._groupwise_factors, dict):
            if self._groupwise_factors_bis and self._task > 0:
                logger.info("Using second set of groupwise lr.")
                groupwise_factor = self._groupwise_factors_bis
            else:
                groupwise_factor = self._groupwise_factors

            params = []
            for group_name, group_params in self._network.get_group_parameters().items():
                if group_params is None or group_name == "last_block":
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if isinstance(factor, list):
                    factor = factor[0] if self._task == 0 else factor[1]
                if factor == 0.:
                    continue
                params.append({"params": group_params, "lr": self._lr * factor})
                logger.info(f"Group: {group_name}, lr: {self._lr * factor}.")
        elif self._groupwise_factors == "ucir":
            params = [
                {
                    "params": self._network.convnet.parameters(),
                    "lr": self._lr
                },
                {
                    "params": self._network.classifier.new_weights,
                    "lr": self._lr
                },
            ]
        else:
            params = self._network.parameters()

        self._optimizer = factory.get_optimizer(params, self._opt_name, self._lr, self._weight_decay)
        
        if self._pretrain_scheduling and self._task == 0:
            self._scheduler = factory.get_lr_scheduler(
                self._pretrain_scheduling, self._optimizer, self._n_pretrain_epochs, lr_decay=self._lr_decay
            )
        else:    
            self._scheduler = factory.get_lr_scheduler(
                self._scheduling, self._optimizer, self._n_epochs, lr_decay=self._lr_decay,
                warmup_config=self._warmup_config, task=self._task
            )

    def _training_step(
        self, train_loader, val_loader, initial_epoch, nb_epochs, record_bn=True, clipper=None
    ):
        best_epoch, best_acc = -1, -1.
        wait = 0

        grad, act = None, None
        if len(self._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._network, self._multiple_devices)
            if self._network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = self._network

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)
            
            # Determine alternate iterations to switch between L^old and L^new (Algorithm 1 Line 6)
            freeze_new = self._task > 0 and \
                            self._class_loss['schedule_old'] > 0 and \
                            epoch >= self._class_loss['start_old'] and \
                            ((epoch+1)%self._class_loss['schedule_old']==0)
            
            # Freeze subset of classifier weights in alternating iterations
            for p in self._network.classifier.new_dim_weights:
                p.requires_grad = not freeze_new 
            for p in self._network.convnet.new_weights.parameters():
                p.requires_grad = not freeze_new 
                
            if self._warmup_config and self._warmup_config['total_epoch'] == epoch+1 and self._reset_classifier:
                self._network.classifier.reset_weights()
                
            if epoch == nb_epochs - 1 and record_bn and len(self._multiple_devices) == 1 and \
               hasattr(training_network.convnet, "record_mode"):
                logger.info("Recording BN means & vars for MCBN...")
                training_network.convnet.clear_records()
                training_network.convnet.record_mode()

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                loss = self._forward_loss(
                    training_network,
                    inputs,
                    targets,
                    memory_flags,
                    gradcam_grad=grad,
                    gradcam_act=act,
                    train_old=freeze_new
                )
                loss.backward()
                self._optimizer.step()

                if clipper:
                    training_network.apply(clipper)
                    
                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._disable_progressbar:
                self._print_metrics(None, epoch, nb_epochs, i)

            if self._scheduler:
                self._scheduler.step(epoch)

            if self._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0:
                self._network.eval()
                self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                ytrue, ypred = self._eval_task(val_loader)
                acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
                logger.info("Val accuracy: {}".format(acc))
                self._network.train()

                if acc > best_acc:
                    best_epoch = epoch
                    best_acc = acc
                    wait = 0
                else:
                    wait += 1

                if self._early_stopping and self._early_stopping["patience"] > wait:
                    logger.warning("Early stopping!")
                    break

        if self._eval_every_x_epochs:
            logger.info("Best accuracy reached at epoch {} with {}%.".format(best_epoch, best_acc))

        if len(self._multiple_devices) == 1 and hasattr(training_network.convnet, "record_mode"):
            training_network.convnet.normal_mode()
            
    def _finetuning_step(
        self, train_loader, val_loader, initial_epoch, nb_epochs, record_bn=True, clipper=None
    ):
        best_epoch, best_acc = -1, -1.
        wait = 0

        grad, act = None, None
        if len(self._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._network, self._multiple_devices)
            if self._network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = self._network

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)
            
            if epoch == nb_epochs - 1 and record_bn and len(self._multiple_devices) == 1 and \
               hasattr(training_network.convnet, "record_mode"):
                logger.info("Recording BN means & vars for MCBN...")
                training_network.convnet.clear_records()
                training_network.convnet.record_mode()

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs, targets = input_dict["inputs"].to(self._device), input_dict["targets"].to(self._device)

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                
                outputs = self._network(inputs)
                post_logits = self._network.post_process(outputs["logits"])
                loss = F.cross_entropy(post_logits/self._finetuning_config.get('temperature', 1), targets)
                loss.backward()
                self._optimizer.step()

                self._metrics["clf"] += loss.item()   
                
                if clipper:
                    training_network.apply(clipper)
                    
                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._disable_progressbar:
                self._print_metrics(None, epoch, nb_epochs, i)

            if self._scheduler:
                self._scheduler.step(epoch)

        if len(self._multiple_devices) == 1 and hasattr(training_network.convnet, "record_mode"):
            training_network.convnet.normal_mode()
            
    def _train_task(self, train_loader, val_loader):
        # Freeeze old weights in feature extractor
        if self._task > 0:
            self._network.freeze(trainable=False, model="convnet_shared")
            self._network.freeze(trainable=False, model="convnet_old")
                    
        for n, p in self._network.named_parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        self._training_step(train_loader, val_loader, 0, self._n_epochs if self._task>0 else self._n_pretrain_epochs)

        # Two-stage learning; fine-tune classifier
        if self._finetuning_config and self._task != 0:
            logger.info("Fine-tuning")
            
            self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                self.inc_dataset, self._herding_indexes
            )
            loader = self.inc_dataset.get_memory_loader(*self.get_memory())

            if self._reset_classifier:
                self._network.classifier.reset_weights()
            
            if self._finetuning_config["tuning"] == "all":
                parameters = self._network.parameters()
            elif self._finetuning_config["tuning"] == "convnet":
                parameters = self._network.convnet.parameters()
            elif self._finetuning_config["tuning"] == "classifier":
                parameters = self._network.classifier.parameters()
            elif self._finetuning_config["tuning"] == "classifier_scale":
                parameters = [
                    {
                        "params": self._network.classifier.parameters(),
                        "lr": self._finetuning_config["lr"]
                    }, {
                        "params": self._network.post_processor.parameters(),
                        "lr": self._finetuning_config["lr"]
                    }
                ]
            else:
                raise NotImplementedError(
                    "Unknwown finetuning parameters {}.".format(self._finetuning_config["tuning"])
                )

            self._optimizer = factory.get_optimizer(
                parameters, self._opt_name, self._finetuning_config["lr"], self._weight_decay
            )
            if self._finetuning_config.get('scheduling'):
                self._scheduler = factory.get_lr_scheduler(
                    self._finetuning_config['scheduling'], self._optimizer, 
                    self._finetuning_config["epochs"], lr_decay=self._finetuning_config['scheduling']
                )
            else:
                self._scheduler = None
                                        
            self._finetuning_step(
                loader,
                val_loader,
                0,
                self._finetuning_config["epochs"],
                record_bn=False
            )

    def _forward_loss(
        self,
        training_network,
        inputs,
        targets,
        memory_flags,
        gradcam_grad=None,
        gradcam_act=None,
        train_old=False,
        **kwargs
    ):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        outputs = training_network(inputs)
        if gradcam_act is not None:
            outputs["gradcam_gradients"] = gradcam_grad
            outputs["gradcam_activations"] = gradcam_act

        loss = self._compute_loss(inputs, outputs, targets, onehot_targets, memory_flags, train_old)

        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss
            
    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags, train_old):
        features, logits, aux_logits = outputs["raw_features"], outputs["logits"], outputs["aux_logits"]

        post_logits = self._network.post_process(logits)
        # Classification loss 
        if self._task == 0 and self._class_loss.get('gamma_pretrain') is not None:
            if self._class_loss['gamma_pretrain'] == 0:
                loss = F.cross_entropy(post_logits, targets, weight=self._class_loss['alpha_pretrain'])
            else:
                loss = losses.FocalLoss(gamma=self._class_loss['gamma_pretrain'], alpha=self._class_loss['alpha_pretrain'])(post_logits, targets)   
        elif self._task > 0 and train_old:
            if self._class_loss['gamma_old'] == 0:
                loss = F.cross_entropy(post_logits, targets, weight=self._class_loss['alpha_old'])
            else:
                loss = losses.FocalLoss(gamma=self._class_loss['gamma_old'], alpha=self._class_loss['alpha_old'])(post_logits, targets)   
        else:
            if self._class_loss['gamma'] == 0:
                loss = F.cross_entropy(post_logits, targets, weight=self._class_loss['alpha'])
            else:
                loss = losses.FocalLoss(gamma=self._class_loss['gamma'], alpha=self._class_loss['alpha'])(post_logits, targets)         
        self._metrics["clf"] += loss.item()

        if self._old_model is not None:
            with torch.no_grad():
                old_outputs = self._old_model(inputs)
                old_features = old_outputs["raw_features"]
                old_logits = self._old_model.post_process(old_outputs["logits"])

            # Auxiliary loss 
            if self._aux_loss['factor'] > 0 and not train_old:
                aux_targets = targets.clone()
                if self._aux_loss['n+1']:
                    aux_targets[targets < self._n_old_classes] = 0
                    aux_targets[targets >= self._n_old_classes] -= self._n_old_classes - 1
                if self._aux_loss['gamma'] == 0:
                    aux_loss = F.cross_entropy(aux_logits, aux_targets, weight=self._aux_loss['alpha']) 
                else:
                    aux_loss = losses.FocalLoss(gamma=self._aux_loss['gamma'], alpha=self._aux_loss['alpha'])(aux_logits, aux_targets)
                aux_loss *= self._aux_loss['factor']
                self._metrics["aux"] += aux_loss.item()        
                loss += aux_loss      

            # Distillation loss 
            if self._dist_loss['factor'] > 0:    
                dist_loss = nn.KLDivLoss()( \
                    F.log_softmax(post_logits[:,:self._n_old_classes]/self._dist_loss['T'], dim=1), \
                    F.softmax(old_logits.detach()/self._dist_loss['T'], dim=1)) * \
                    self._dist_loss['T'] * self._dist_loss['T'] * \
                    self._dist_loss['factor'] * self._n_old_classes   
                self._metrics["dist"] += dist_loss.item()
                loss += dist_loss            
                
            # Margin loss 
            if self._ranking_loss['factor'] > 0:
                ranking_loss = self._ranking_loss["factor"] * losses.ucir_ranking(
                    logits,
                    targets,
                    self._n_classes,
                    self._task_size,
                    nb_negatives=min(self._ranking_loss["nb_negatives"], self._task_size),
                    margin=self._ranking_loss["margin"]
                )
                loss += ranking_loss
                self._metrics["rank"] += ranking_loss.item()

        return loss

    def build_examplars(
        self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train"
    ):
        logger.info("Building & updating memory.")
        memory_per_class = memory_per_class or self._memory_per_class
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, targets_memory = [], []
        class_means = np.zeros((self._n_classes, self._network.features_dim * (self._task+1)))

        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source
            )
            features, targets = utils.extract_features(self._network, loader)
            features_flipped, _ = utils.extract_features(
                self._network,
                inc_dataset.get_custom_loader(class_idx, mode="flip", data_source=data_source)[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                if self._herding_selection["type"] == "icarl":
                    selected_indexes = herding.icarl_selection(features, memory_per_class)
                elif self._herding_selection["type"] == "closest":
                    selected_indexes = herding.closest_to_mean(features, memory_per_class)
                elif self._herding_selection["type"] == "random":
                    selected_indexes = herding.random(features, memory_per_class)
                elif self._herding_selection["type"] == "first":
                    selected_indexes = np.arange(memory_per_class)
                elif self._herding_selection["type"] == "kmeans":
                    selected_indexes = herding.kmeans(
                        features, memory_per_class, k=self._herding_selection["k"]
                    )
                elif self._herding_selection["type"] == "confusion":
                    selected_indexes = herding.confusion(
                        *self._last_results,
                        memory_per_class,
                        class_id=class_idx,
                        minimize_confusion=self._herding_selection["minimize_confusion"]
                    )
                elif self._herding_selection["type"] == "var_ratio":
                    selected_indexes = herding.var_ratio(
                        memory_per_class, self._network, loader, **self._herding_selection
                    )
                elif self._herding_selection["type"] == "mcbn":
                    selected_indexes = herding.mcbn(
                        memory_per_class, self._network, loader, **self._herding_selection
                    )
                else:
                    raise ValueError(
                        "Unknown herding selection {}.".format(self._herding_selection)
                    )

                herding_indexes.append(selected_indexes)

            # Reducing examplars:
            try:
                selected_indexes = herding_indexes[class_idx][:memory_per_class]
                herding_indexes[class_idx] = selected_indexes
            except:
                import pdb
                pdb.set_trace()

            # Re-computing the examplar mean (which may have changed due to the training):
            examplar_mean = self.compute_examplar_mean(
                features, features_flipped, selected_indexes, memory_per_class
            )

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])

            class_means[class_idx, :] = examplar_mean

        data_memory = np.concatenate(data_memory)
        targets_memory = np.concatenate(targets_memory)

        return data_memory, targets_memory, herding_indexes, class_means
    