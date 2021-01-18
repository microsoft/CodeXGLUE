import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from transformers.modeling_bert import BertLayerNorm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           BertConfig, BertForMaskedLM, BertTokenizer,
#                           GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
#                           OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                           RobertaConfig, RobertaModel, RobertaTokenizer,
#                           DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from transformers.modeling_utils import PreTrainedModel


class Model(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 1))
        loss = self.loss_func(logits, labels.float())
        predictions = (logits > 0.5).int()  # (Batch, )
        return loss, predictions

