import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils import class_weight
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, CamembertConfig, CamembertModel,
                          DataCollator, EvalPrediction, HfArgumentParser,
                          PreTrainedModel, PreTrainedTokenizer, RobertaConfig,
                          RobertaModel, Trainer, TrainingArguments, set_seed)
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_roberta import *
from transformers.trainer import get_tpu_sampler
from transformers.training_args import is_tpu_available

from torchsampler import ImbalancedDatasetSampler, ImbalancedDatasetSamplerVec, MultilabelBalancedRandomSampler
from utils.misc import input_columns, target_columns
import ast
from transformers.modeling_roberta import RobertaModel, RobertaClassificationHead

from transformers.modeling_camembert import (
    CAMEMBERT_START_DOCSTRING,
)

set_seed(1029)

os.environ["WANDB_API_KEY"] = "d3551f5d511452da6512eceb1a03fe3caad643d2"

wandb.init(project="speed_training_camembert_gravite")



class MyCamembertWeighted(BertPreTrainedModel):


    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()

                class_weights = None
                try:
                    class_weights = self.class_weights
                except AttributeError:
                    pass
                if class_weights is not None:
                    loss_fct = CrossEntropyLoss(weight=class_weights)
                
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class CustomMultilabel(BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        super(CustomMultilabel, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = CamembertModel(config)
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_layers = outputs[2]

        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )

        outputs = logits

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            
            # Set class weights
            class_weights = None
            try:
                class_weights = self.class_weights
            except AttributeError:
                pass
            if class_weights is not None:
                loss_fct = BCEWithLogitsLoss(weight=class_weights)

            loss = loss_fct(logits, labels)

            outputs = (loss, logits)

        return outputs


class RobertaForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Roberta model adapted for multi-label sequence classification
    """

    config_class = CamembertConfig
    base_model_prefix = "camembert"

    def __init__(self, config, pos_weight=None):
        super(RobertaForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.roberta = CamembertModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs
  

class MyTrainer(Trainer):

    """
    Modified Trainer for balanced sampling option
    """

    def __init__(self, balanced_sampler, multilabel, **kwds):
        super().__init__(**kwds)
        self.balanced_sampler = balanced_sampler
        self.multilabel = multilabel

    def _setup_wandb(self):
        wandb.init(project="speed_training_camembert_gravite",
                   config=vars(self.args),
                   name=self.args.output_dir)
        wandb.watch(self.model, log="gradients", log_freq=self.args.logging_steps)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:

            if self.balanced_sampler:

                if self.multilabel:
                    train_sampler = MultilabelBalancedRandomSampler(self.train_dataset)
                else:
                    train_sampler = ImbalancedDatasetSampler(self.train_dataset)
                   
            else:
                train_sampler = (
                    RandomSampler(self.train_dataset)
                    if self.args.local_rank == -1
                    else DistributedSampler(self.train_dataset)
                )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader


@dataclass
class Example:
    text: str
    label: list


@dataclass
class Features:
    input_ids: List[int]
    attention_mask: List[int]
    label: int


@dataclass
class ModelParameters:
    max_seq_len: Optional[int] = field(
        default=None,
        metadata={"help": "max seq len"},
    )
    dynamic_padding: bool = field(
        default=False,
        metadata={"help": "limit pad size at batch level"},
    )
    smart_batching: bool = field(
        default=False,
        metadata={"help": "build batch of similar sizes"},
    )
    balanced_sampler: bool = field(
        default=False,
        metadata={"help": "class balancing in batches"},
    )
    class_weights: bool = field(
        default=False,
        metadata={"help": "class weighting"},
    )
    multilabel: bool = field(
        default=False,
        metadata={"help": "define multilabel problem"},
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "define the number of labels"},
    )



class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_max_length: bool, max_len: int,
                 examples: List[Example]) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: List[Example] = examples
        self.current = 0
        self.pad_to_max_length = pad_to_max_length

    def encode(self, ex: Example) -> Features:
        encode_dict = self.tokenizer.encode_plus(text=ex.text,
                                                 add_special_tokens=True,
                                                 max_length=self.max_len,
                                                 pad_to_max_length=self.pad_to_max_length,
                                                 return_token_type_ids=False,
                                                 return_attention_mask=True,
                                                 return_overflowing_tokens=False,
                                                 return_special_tokens_mask=False,
                                                 )
        return Features(input_ids=encode_dict["input_ids"],
                        attention_mask=encode_dict["attention_mask"],
                        label=ex.label)

    def __getitem__(self, idx) -> Features:  # Trainer doesn't support IterableDataset (define sampler)
        if self.current == len(self.examples):
            self.current = 0

        ex = (self.examples[idx] if model_args.balanced_sampler else
              self.examples[idx])   # idx to make balanced sampling work  self.examples[self.current])
        self.current += 1
        return self.encode(ex=ex)

    def __len__(self):
        return len(self.examples)


def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]


@dataclass
class SmartCollator(DataCollator):
    pad_token_id: int

    def collate_batch(self, batch: List[Features]) -> Dict[str, torch.Tensor]:
        batch_inputs = list()
        batch_attention_masks = list()
        labels = list()
        max_size = max([len(ex.input_ids) for ex in batch])
        for item in batch:
            batch_inputs += [pad_seq(item.input_ids, max_size, self.pad_token_id)]
            batch_attention_masks += [pad_seq(item.attention_mask, max_size, 0)]
            labels.append(item.label)

        return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.float) # float to support multilabel
                }


def load_transformers_model(pretrained_model_name_or_path: str,
                            use_cuda: bool,
                            mixed_precision: bool,
                            class_weights: np.array,
                            multilabel: bool,
                            num_labels: int) -> PreTrainedModel:

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                        num_labels=num_labels)

    if class_weights is None:

        if not multilabel:
            print("Non weighted model")
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                config=config)
        else:
            print("Non weighted multilabel model")
            model = RobertaForMultiLabelSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                config=config)
    else:
        if not multilabel:
            print("Weighted model")
            model = MyCamembertWeighted.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                config=config)
        else:
            print("Weighted multilabel model")
            # test implement multilabel with weights
            model = CustomMultilabel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                config=config)

        # Set model weights
        model.set_class_weights(class_weights) 

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)

    if mixed_precision:
        print("MIXED PRECISION")
        try:
            from apex import amp
            model = amp.initialize(model, opt_level='O1')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    return model


def load_train_data(path: str, sort: bool) -> List[Example]:
    
    sentences = list()
    x_train = np.load(path+'X_train.npy', allow_pickle=True).tolist()
    y_train = np.load(path+'y_train.npy', allow_pickle=True).tolist()

    for train, label in zip(x_train, y_train):
        lab = len(train)
        sentences.append((lab, Example(text=train, label=label)))
    if sort:
        sentences.sort(key=lambda x: x[0])

    return [e for (_, e) in sentences]


def load_dev_data(path: str) -> List[Example]:
    sentences = list()
    x_test = np.load(path+'X_test.npy', allow_pickle=True).tolist()
    y_test = np.load(path+'y_test.npy', allow_pickle=True).tolist()

    for test, label in zip(x_test, y_test):
        lab = len(test)
        sentences.append((lab, Example(text=test, label=label)))

    sentences.sort(key=lambda x: x[0])

    return [e for (_, e) in sentences]


def build_batches(sentences: List[Example], batch_size: int) -> List[Example]:
    batch_ordered_sentences = list()
    while len(sentences) > 0:
        to_take = min(batch_size, len(sentences))
        select = random.randint(0, len(sentences) - to_take)
        batch_ordered_sentences += sentences[select:select + to_take]
        del sentences[select:select + to_take]
    return batch_ordered_sentences


def compute_metrics(p: EvalPrediction) -> Dict:
    # implement different metric between multilabel 
    preds = np.argmax(p.predictions, axis=1)
    print(preds)
    accuracy = accuracy_score(p.label_ids, preds)
    balanced_accuracy = balanced_accuracy_score(p.label_ids, preds)
    return {"acc": accuracy, "b_acc": balanced_accuracy}

def compute_metrics_multi(p: EvalPrediction) -> Dict:
    def accuracy_thresh(y_true, y_pred, thresh: float=0.5):
        return ((y_pred > thresh) == y_true).mean()
    
    print(p.predictions)
    preds = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    accuracy = accuracy_thresh(p.label_ids, preds)
    roc_auc = roc_auc_score(p.label_ids, preds, average="samples")
    my_preds_label = (preds > 0.5).astype(int)
    my_preds_true = p.label_ids
    precision  = precision_score(my_preds_true, my_preds_label, average="samples")
    recall  = recall_score(my_preds_true, my_preds_label, average="samples")
    f1_sample = f1_score(my_preds_true, my_preds_label, average="samples")
    return {"prec": precision, "rec": recall,"f1_sample": f1_sample, "roc":roc_auc}


def build_compute_metrics_fn(multilabel: bool) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        if multilabel:
            metric = compute_metrics_multi(p)
        else:
            metric = compute_metrics(p)

        return metric

    return compute_metrics_fn


if __name__ == "__main__":

    parser = HfArgumentParser((TrainingArguments, ModelParameters))
    training_args, model_args = parser.parse_args_into_dataclasses()  # type: (TrainingArguments, ModelParameters)

    print(training_args, model_args)

    # Define class weights
    if model_args.class_weights:
        class_weights = [1,1,1,20]
        y = pd.read_csv("data/train_test/train.csv", usecols=['TEF_ID'], sep='|')
        class_weights = class_weight.compute_class_weight('balanced', y['TEF_ID'].unique(), y['TEF_ID'].values)
        class_weights = torch.tensor(class_weights, dtype=torch.float, device=training_args.device)

    else:
        class_weights = None

    train_sentences = load_train_data(path="data/train_test/",
                                      sort=model_args.smart_batching)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="camembert-base")
    if model_args.max_seq_len:
        max_sequence_len = model_args.max_seq_len
    else:
        longest_sentence = max(train_sentences, key=len)
        max_sequence_len = len(tokenizer.encode(text=longest_sentence.text))

    train_batches = build_batches(sentences=train_sentences, batch_size=training_args.per_gpu_train_batch_size)
    valid_sentences = load_dev_data(path="data/train_test/")
    valid_batches = build_batches(sentences=valid_sentences, batch_size=training_args.per_gpu_eval_batch_size)

    train_set = TextDataset(tokenizer=tokenizer,
                            max_len=max_sequence_len,
                            examples=train_batches,
                            pad_to_max_length=not model_args.dynamic_padding)

    valid_set = TextDataset(tokenizer=tokenizer,
                            max_len=max_sequence_len,
                            examples=valid_batches,
                            pad_to_max_length=not model_args.dynamic_padding)
                     
    model = load_transformers_model(pretrained_model_name_or_path="camembert-base",
                                    use_cuda=True,
                                    mixed_precision=False,
                                    class_weights=class_weights,
                                    multilabel=model_args.multilabel,
                                    num_labels=model_args.num_labels)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        data_collator=SmartCollator(pad_token_id=tokenizer.pad_token_id),
        tb_writer=SummaryWriter(log_dir='logs', flush_secs=10),
        eval_dataset=valid_set,
        compute_metrics=build_compute_metrics_fn(model_args.multilabel),
        balanced_sampler=model_args.balanced_sampler,
        multilabel=model_args.multilabel
    )

    start_time = time.time()
    trainer.train()
    wandb.config.update(model_args)
    wandb.config.update(training_args)
    wandb.log({"training time": int((time.time() - start_time) / 60)})
    trainer.save_model()
    #trainer.evaluate()
    logging.info("*** Evaluate ***")
    result = trainer.evaluate()
    wandb.log(result)

    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logging.info("***** Eval results *****")
        for key, value in result.items():
            logging.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))
