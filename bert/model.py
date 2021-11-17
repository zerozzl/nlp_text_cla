import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class Bert(nn.Module):
    def __init__(self, config_path, model_path, num_tags, bert_freeze):
        super(Bert, self).__init__()

        config = BertConfig.from_json_file(config_path)
        self.embedding = BertModel.from_pretrained(model_path, config=config)
        self.linear = nn.Linear(config.hidden_size, num_tags)

        if bert_freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, tokens, segments, masks, decode=True, tags=None):
        out = self.embedding(input_ids=tokens, token_type_ids=segments, attention_mask=masks)
        out = out.last_hidden_state
        out = out[:, 0, :]
        out = self.linear(out)

        if decode:
            out = F.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)
            pred = out.cpu().numpy()
            return pred
        else:
            loss = self.ce_loss(out, tags)
            return loss
