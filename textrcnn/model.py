import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self, num_tags, vocab_size, embed_size, hidden_size,
                 input_dropout_rate, hidden_dropout_rate, embed_type='rand'):
        super(TextRCNN, self).__init__()
        self.embed_type = embed_type

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embed_type == 'static':
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.rnn = nn.LSTM(input_size=embed_size,
                           hidden_size=embed_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)

        self.linear_rep = nn.Linear(embed_size * 3, hidden_size)
        self.linear_out = nn.Linear(hidden_size, num_tags)

        self.in_dropout = nn.Dropout(input_dropout_rate)
        self.hid_dropout = nn.Dropout(hidden_dropout_rate)

        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, tokens, masks, decode=True, tags=None):
        # tokens_len = torch.sum(masks, 1).cpu()
        masks = masks.unsqueeze(2)

        embed = self.embedding(tokens)
        # embed_ctx = rnn.pack_padded_sequence(embed, tokens_len, batch_first=True)
        embed_ctx, _ = self.rnn(embed)
        # embed_ctx, _ = rnn.pad_packed_sequence(embed_ctx, batch_first=True)

        embed_com = torch.cat((embed, embed_ctx), dim=2)
        embed_com = torch.mul(embed_com, masks)
        embed_com = self.in_dropout(embed_com)

        ctx = F.relu(self.linear_rep(embed_com))
        ctx = ctx.permute(0, 2, 1)
        ctx = torch.max_pool1d(ctx, ctx.size(2)).squeeze(2)
        ctx = self.hid_dropout(ctx)

        out = self.linear_out(ctx)

        if decode:
            out = F.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)
            pred = out.cpu().numpy()
            return pred
        else:
            loss = self.ce_loss(out, tags)
            return loss
