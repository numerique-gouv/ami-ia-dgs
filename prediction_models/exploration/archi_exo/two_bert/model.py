import logging
from transformers import BertModel, CamembertModel
from transformers.modeling_bert import BertPreTrainedModel
import torch
from torch import nn

logging.getLogger("transformers").setLevel(logging.ERROR)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class CustomRobertaSiam(BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        super(CustomRobertaSiam, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = CamembertModel(config)
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids_text1,
        attention_mask_text1,
        token_type_ids_text1,
        input_ids_text2,
        attention_mask_text2,
        token_type_ids_text2
    ):
        seg_text1_ids = torch.zeros_like(token_type_ids_text1)
        outputs_text1 = self.roberta(
            input_ids_text1,
            attention_mask=attention_mask_text1,
            token_type_ids=seg_text1_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )

        hidden_layers_text1 = outputs_text1[2]

        cls_outputs_text1 = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers_text1], dim=2
        )
        cls_output_text1 = (torch.softmax(self.layer_weights, dim=0) * cls_outputs_text1).sum(-1)

        seg_text2_ids = torch.zeros_like(token_type_ids_text2)
        outputs_text2 = self.roberta(
            input_ids_text2,
            attention_mask=attention_mask_text2,
            token_type_ids=seg_text2_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )

        hidden_layers_text2 = outputs_text2[2]

        cls_outputs_text2 = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers_text2], dim=2
        )
        cls_output_text2 = (torch.softmax(self.layer_weights, dim=0) * cls_outputs_text2).sum(-1)

        cls_output = torch.cat([cls_output_text1, cls_output_text2 ], dim=1)

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )

        outputs = logits
        return outputs


class CustomRobertaTwo(BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        super(CustomRobertaTwo, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta_text1 = CamembertModel(config)
        self.roberta_text2 = CamembertModel(config)
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids_text1,
            attention_mask_text1,
            token_type_ids_text1,
            input_ids_text2,
            attention_mask_text2,
            token_type_ids_text2
    ):
        seg_text1_ids = torch.zeros_like(token_type_ids_text1)
        outputs_text1 = self.roberta_text1(
            input_ids_text1,
            attention_mask=attention_mask_text1,
            token_type_ids=seg_text1_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )

        hidden_layers_text1 = outputs_text1[2]

        cls_outputs_text1 = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers_text1], dim=2
        )
        cls_output_text1 = (torch.softmax(self.layer_weights, dim=0) * cls_outputs_text1).sum(-1)

        seg_text2_ids = torch.zeros_like(token_type_ids_text2)
        outputs_text2 = self.roberta_text2(
            input_ids_text2,
            attention_mask=attention_mask_text2,
            token_type_ids=seg_text2_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )

        hidden_layers_text2 = outputs_text2[2]

        cls_outputs_text2 = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers_text2], dim=2
        )
        cls_output_text2 = (torch.softmax(self.layer_weights, dim=0) * cls_outputs_text2).sum(-1)

        cls_output = torch.cat([cls_output_text1, cls_output_text2], dim=1)

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )

        outputs = logits
        return outputs


def get_model_optimizer(args):
    if args.model_type == "roberta_two":
        model = CustomRobertaTwo.from_pretrained(args.bert_model, num_labels=args.num_classes)
        prefix = "roberta_two"
    elif args.model_type == "roberta_siam":
        model = CustomRobertaSiam.from_pretrained(
            args.bert_model, num_labels=args.num_classes
        )
        prefix = "roberta_siam"
    else:
        raise ValueError("Wrong model_type {}".format(args.model_type))

    model.cuda()
    model = nn.DataParallel(model)
    params = list(model.named_parameters())

    def is_backbone(n):
        return prefix in n

    optimizer_grouped_parameters = [
        {"params": [p for n, p in params if is_backbone(n)], "lr": args.lr},
        {"params": [p for n, p in params if not is_backbone(n)], "lr": args.lr * 500},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.lr, weight_decay=0
    )

    return model, optimizer
