import torch
from torch import nn
import torch.nn.functional as F


class FastText(nn.Module):
    def __init__(self, num_tags, vocab_size, embed_size, input_dropout_rate,
                 embed_type='rand', use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0):
        super(FastText, self).__init__()
        self.embed_type = embed_type
        self.use_bigram = use_bigram

        hidden_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        if use_bigram:
            hidden_size += bigram_embed_size * 2
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_embed_size)

        if embed_type == 'static':
            for param in self.embedding.parameters():
                param.requires_grad = False

            if use_bigram:
                for param in self.bigram_embedding.parameters():
                    param.requires_grad = False

        self.linear = nn.Linear(hidden_size, num_tags)

        self.in_dropout = nn.Dropout(input_dropout_rate)

        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, tokens, masks, bigram, decode=True, tags=None):
        masks = masks.unsqueeze(2)

        embed = self.embedding(tokens)
        if self.use_bigram:
            embed_bi = torch.cat([self.bigram_embedding(bigram[:, :, i]) for i in range(bigram.size()[2])], dim=2)
            embed = torch.cat((embed, embed_bi), dim=2)
        embed = self.in_dropout(embed)

        context = torch.mul(embed, masks)
        context = context.mean(1)

        out = self.linear(context)

        if decode:
            out = F.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)
            pred = out.cpu().numpy()
            return pred
        else:
            loss = self.ce_loss(out, tags)
            return loss
