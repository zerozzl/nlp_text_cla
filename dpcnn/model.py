import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, kernel_num):
        super(ResidualBlock, self).__init__()

        self.pooling = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(kernel_num, kernel_num, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(kernel_num, kernel_num, kernel_size=3, padding=1))
        )

    def forward(self, x):
        pool = self.pooling(x)
        out = self.conv_block(pool)
        out = torch.add(pool, out)
        return out


class DPCNN(nn.Module):
    def __init__(self, num_tags, vocab_size, embed_size, input_size, kernel_num,
                 input_dropout_rate, hidden_dropout_rate, embed_type='rand'):
        super(DPCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embed_type == 'static':
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.region_embedding = nn.utils.weight_norm(nn.Conv1d(embed_size, kernel_num, kernel_size=3, padding=1))

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(kernel_num, kernel_num, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(kernel_num, kernel_num, kernel_size=3, padding=1))
        )

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(kernel_num) for _ in range(int(np.log2(input_size)))]
        )

        self.linear = nn.Linear(kernel_num, num_tags)

        self.in_dropout = nn.Dropout(input_dropout_rate)
        self.hid_dropout = nn.Dropout(hidden_dropout_rate)

        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, tokens, decode=True, tags=None):
        embed = self.embedding(tokens)
        embed = self.in_dropout(embed)
        embed = embed.permute(0, 2, 1)

        region = self.region_embedding(embed)
        out = self.conv_block(region)
        out = torch.add(region, out)

        for block in self.residual_blocks:
            out = block(out)
        out = out.squeeze(2)
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
