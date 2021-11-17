import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn


class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLayer, self).__init__()
        self.w = nn.Linear(input_size, hidden_size)
        self.u = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.tanh(self.w(x))
        out = F.softmax(self.u(out), dim=1)
        out = out.mul(x).sum(1)
        return out


class HAN(nn.Module):
    def __init__(self, num_tags, vocab_size, embed_size, hidden_size,
                 input_dropout_rate, hidden_dropout_rate, embed_type='rand'):
        super(HAN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embed_type == 'static':
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.word_encoder = nn.GRU(input_size=embed_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   bidirectional=True)
        self.word_attention = AttentionLayer(hidden_size * 2, hidden_size)

        self.sent_encoder = nn.GRU(input_size=hidden_size * 2,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   bidirectional=True)
        self.sent_attention = AttentionLayer(hidden_size * 2, hidden_size)

        self.linear = nn.Linear(hidden_size * 2, num_tags)

        self.in_dropout = nn.Dropout(input_dropout_rate)
        self.hid_dropout = nn.Dropout(hidden_dropout_rate)

        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, sents, decode=True, tags=None):
        sents_len = [len(sent) for sent in sents]

        contexts = []
        for i in range(len(sents)):
            embed = self.embedding(sents[i])
            embed = self.in_dropout(embed)

            context, _ = self.word_encoder(embed)
            context = self.word_attention(context)

            contexts.append(context)

        contexts = rnn.pad_sequence(contexts, batch_first=True)
        contexts = rnn.pack_padded_sequence(contexts, sents_len, batch_first=True)
        contexts, _ = self.sent_encoder(contexts)
        contexts, _ = rnn.pad_packed_sequence(contexts, batch_first=True)
        contexts = self.sent_attention(contexts)
        contexts = self.hid_dropout(contexts)

        out = self.linear(contexts)

        if decode:
            out = F.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)
            pred = out.cpu().numpy()
            return pred
        else:
            loss = self.ce_loss(out, tags)
            return loss
