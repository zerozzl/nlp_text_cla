import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, num_tags, vocab_size, embed_size, kernel_sizes, kernel_num,
                 input_dropout_rate, hidden_dropout_rate, embed_type='rand'):
        super(TextCNN, self).__init__()
        self.embed_type = embed_type

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embed_type == 'static':
            for param in self.embedding.parameters():
                param.requires_grad = False
        elif embed_type == 'multi':
            self.embedding_ext = nn.Embedding(vocab_size, embed_size)
            for param in self.embedding_ext.parameters():
                param.requires_grad = False

        kernel_sizes = [int(x) for x in kernel_sizes.split(',')]
        self.convs = nn.ModuleList(
            [nn.utils.weight_norm(nn.Conv1d(embed_size, kernel_num, ks, padding=int(ks / 2))) for ks in kernel_sizes]
        )

        self.linear = nn.Linear(len(kernel_sizes) * kernel_num, num_tags)

        self.in_dropout = nn.Dropout(input_dropout_rate)
        self.hid_dropout = nn.Dropout(hidden_dropout_rate)

        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_embedding_ext(self, pretrained_embeddings):
        self.embedding_ext.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, tokens, decode=True, tags=None):
        embed = self.embedding(tokens)
        embed = self.in_dropout(embed)
        embed = embed.permute(0, 2, 1)

        if self.embed_type == 'multi':
            embed_ext = self.embedding_ext(tokens)
            embed_ext = self.in_dropout(embed_ext)
            embed_ext = embed_ext.permute(0, 2, 1)

        out = []
        for conv in self.convs:
            conv_out = F.relu(conv(embed))
            if self.embed_type == 'multi':
                conv_out_ext = F.relu(conv(embed_ext))
                conv_out = torch.add(conv_out, conv_out_ext)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            out.append(conv_out)
        out = torch.cat(out, dim=1)
        out = self.hid_dropout(out)

        out = self.linear(out)

        if decode:
            out = F.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)
            pred = out.cpu().numpy()
            return pred
        else:
            loss = self.ce_loss(out, tags)
            return loss
