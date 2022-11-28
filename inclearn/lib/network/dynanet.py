import copy
import logging

import torch
from torch import nn
import torch.nn.functional as F

from inclearn.lib import factory

from .classifiers import (DynaClassifier, DynaCosineClassifier)
from .postprocessors import FactorScalar, HeatedUpScalar, InvertedFactorScalar
from .word import Word2vec

logger = logging.getLogger(__name__)


class DynamicNet(nn.Module):

    def __init__(
        self,
        convnet_type,
        convnet_kwargs={},
        classifier_kwargs={},
        aux_classifier_kwargs={},
        postprocessor_kwargs={},
        init="kaiming",
        device=None,
        return_features=False,
        extract_no_act=False,
        classifier_no_act=False
    ):
        super(DynamicNet, self).__init__()

        if postprocessor_kwargs.get("type") == "learned_scaling":
            self.post_processor = FactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "inverted_learned_scaling":
            self.post_processor = InvertedFactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "heatedup":
            self.post_processor = HeatedUpScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") is None:
            self.post_processor = None
        else:
            raise NotImplementedError(
                "Unknown postprocessor {}.".format(postprocessor_kwargs["type"])
            )
        logger.info("Post processor is: {}".format(self.post_processor))

        self.convnet = factory.get_convnet(convnet_type, **convnet_kwargs)

        if "type" not in classifier_kwargs:
            raise ValueError("Specify a classifier!", classifier_kwargs)
        if classifier_kwargs["type"] == "fc":
            self.classifier = DynaClassifier(self.convnet.out_dim, device=device, **classifier_kwargs)
        elif classifier_kwargs["type"] == "cosine":
            self.classifier = DynaCosineClassifier(
                self.convnet.out_dim, device=device, **classifier_kwargs
            )
        else:
            raise ValueError("Unknown classifier type {}.".format(classifier_kwargs["type"]))

        if aux_classifier_kwargs.get("type") == "fc":
            self.aux_classifier = DynaClassifier(self.convnet.out_dim, device=device, **aux_classifier_kwargs)
            self.aux_post_processor = None
        elif aux_classifier_kwargs.get("type") == "cosine":
            self.aux_classifier = DynaCosineClassifier(
                self.convnet.out_dim, device=device, **aux_classifier_kwargs
            )
            self.aux_post_processor = FactorScalar(**postprocessor_kwargs)
        elif aux_classifier_kwargs.get("type") is None:
            self.aux_classifier = None
            self.aux_post_processor = None
        else:
            raise NotImplementedError(
                "Unknown aux classifier {}.".format(aux_classifier_kwargs["type"])
            )
        logger.info("Aux classifier is: {}".format(self.post_processor))
            
        self.return_features = return_features
        self.extract_no_act = extract_no_act
        self.classifier_no_act = classifier_no_act
        self.device = device
        self.task = 0

        if self.extract_no_act:
            logger.info("Features will be extracted without the last ReLU.")
        if self.classifier_no_act:
            logger.info("No ReLU will be applied on features before feeding the classifier.")

        self.to(self.device)
        
    def on_task_end(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.on_task_end()
        if isinstance(self.post_processor, nn.Module):
            self.post_processor.on_task_end()
        if isinstance(self.aux_classifier, nn.Module):
            self.aux_classifier.on_task_end()
        if isinstance(self.aux_post_processor, nn.Module):
            self.aux_post_processor.on_task_end()

    def on_epoch_end(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.on_epoch_end()
        if isinstance(self.post_processor, nn.Module):
            self.post_processor.on_epoch_end()
        if isinstance(self.aux_classifier, nn.Module):
            self.aux_classifier.on_epoch_end()
        if isinstance(self.aux_post_processor, nn.Module):
            self.aux_post_processor.on_epoch_end()

    def forward(self, x):
        outputs = self.convnet(x)

        if hasattr(self, "classifier_no_act") and self.classifier_no_act:
            selected_features = outputs["raw_features"]
        else:
            selected_features = outputs["features"]

        clf_outputs = self.classifier(selected_features)
        outputs.update(clf_outputs)
        
        if self.aux_classifier is not None and self.task > 1:
            aux_outputs = self.aux_classifier(selected_features[:,-self.features_dim:])
            outputs.update({"aux_logits": self.aux_post_processor(aux_outputs["logits"])})
        elif self.training and self.task > 1:
            aux_outputs = F.linear(F.normalize(selected_features[:,-self.features_dim:], p=2, dim=1), \
                                  F.normalize(self.classifier.weights[:,-self.features_dim:], p=2, dim=1))
            outputs.update({"aux_logits": aux_outputs})
        else:
            outputs.update({"aux_logits": None})

        return outputs
    
    def post_process(self, x):
        if self.post_processor is None:
            return x
        return self.post_processor(x)

    @property
    def features_dim(self):
        return self.convnet.out_dim

    def add_classes(self, n_classes):
        self.task += 1
        self.convnet.add_classes()
        self.classifier.add_classes(n_classes)
        if self.aux_classifier is not None: 
            if self.task == 2:
                self.aux_classifier.add_classes(n_classes+1)
            elif self.task > 2:
                self.aux_classifier.reset_weights()
    
    def add_imprinted_classes(self, class_indexes, inc_dataset, **kwargs):
        self.task += 1
        self.convnet.add_classes()
        if hasattr(self.classifier, "add_imprinted_classes"):
            self.classifier.add_imprinted_classes(class_indexes, inc_dataset, self, **kwargs)
        if self.aux_classifier is not None: 
            if self.task == 2:
                self.aux_classifier.add_classes(len(class_indexes)+1)
            elif self.task > 2:
                self.aux_classifier.reset_weights()

    def add_custom_weights(self, weights, **kwargs):
        self.task += 1
        self.convnet.add_classes()
        self.classifier.add_custom_weights(weights, **kwargs)
        if self.aux_classifier is not None: 
            if self.task == 2:
                self.aux_classifier.add_classes(weights.shape[0]+1)
            elif self.task > 2:
                self.aux_classifier.reset_weights()

    def extract(self, x):
        outputs = self.convnet(x)
        if self.extract_no_act:
            return outputs["raw_features"]
        return outputs["features"]

    def freeze(self, trainable=False, model="all"):
        if model == "all":
            model = self
        elif model == "convnet_old":
            model = self.convnet.old_weights
        elif model == "convnet_shared":
            model = self.convnet.shared_weights
        elif model == "convnet":
            model = self.convnet
        elif model == "classifier_old":
            model = self.classifier.old_weights
        elif model == "classifier":
            model = self.classifier
        else:
            assert False, model

        if not isinstance(model, nn.Module):
            return self

        for name, param in model.named_parameters():
            param.requires_grad = trainable

        if not trainable:
            model.eval()
        else:
            model.train()

        return self

    def get_group_parameters(self):
        groups = {"convnet_new": self.convnet.new_weights.parameters()}
        
        if self.convnet.shared_weights is not None:
            groups["convet_shared"] = self.convnet.shared_weights.parameters()         
        if hasattr(self.convnet, "old_weights") and self.convnet.old_weights is not None:
            groups["convnet_old"] = self.convnet.old_weights.parameters()         
        if isinstance(self.post_processor, FactorScalar):
            groups["postprocessing"] = self.post_processor.parameters()
        if self.aux_classifier:
            groups["aux_weights"] = self.aux_classifier.parameters()
        if isinstance(self.aux_post_processor, FactorScalar):
            groups["aux_postprocessing"] = self.aux_post_processor.parameters()
        if hasattr(self.classifier, "new_cls_weights"):
            groups["new_cls_weights"] = self.classifier.new_cls_weights
        if hasattr(self.classifier, "new_dim_weights"):
            groups["new_dim_weights"] = self.classifier.new_dim_weights
        if hasattr(self.classifier, "old_weights"):
            groups["old_weights"] = self.classifier.old_weights
        if hasattr(self.convnet, "last_block"):
            groups["last_block"] = self.convnet.last_block.parameters()
        if hasattr(self.classifier, "_negative_weights"
                  ) and isinstance(self.classifier._negative_weights, nn.Parameter):
            groups["neg_weights"] = self.classifier._negative_weights

        return groups

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_classes(self):
        return self.classifier.n_classes
