import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchsummary import summary
import math


def train_val_dataset(dataset, val_split=0.15):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    return datasets


def attention(q, k, v, d_k, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class Transformer(nn.Module):
    def __init__(self, seq_len, channels, d_model, heads, n_classes, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(channels, d_model)
        self.norm = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(seq_len*d_model, n_classes)

    def forward(self, x):
        x = self.linear1(x)
        x2 = self.norm(x)
        x = x + self.dropout(self.attn(x2, x2, x2))
        x2 = self.norm(x)
        x = x + self.dropout(self.ff(x2))
        x = torch.flatten(x, start_dim=1)
        out = self.linear2(x)
        return out


class GaussianDistribution(Dataset):

    def __init__(self, seq_len, channels):
        self.channels = channels
        self.seq_len = seq_len
        first_cluster = torch.empty(size=(1000, seq_len, channels)).normal_(mean=5, std=0.5)
        first_labels = torch.zeros(1000, dtype=torch.int64)
        second_cluster = torch.empty(size=(1000, seq_len, channels)).normal_(mean=10, std=0.5)
        second_labels = torch.ones(1000, dtype=torch.int64)
        self.x = torch.cat(tensors=(first_cluster, second_cluster))
        self.y = torch.cat(tensors=(first_labels, second_labels))
        self.n_samples = self.x.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 10
    channels = 2
    batch_size = 8
    num_epochs = 20
    learning_rate = 0.001

    dataset_full = GaussianDistribution(seq_len=seq_len, channels=channels)

    dataset = train_val_dataset(dataset_full)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}
    dataiter = iter(train_loader)
    samples, labels = dataiter.next()
    #print(samples[0:2])
    print(samples.shape)

    model = Transformer(seq_len=seq_len, channels=channels, d_model=512, heads=1, n_classes=2)
    # if gpu
    # model.to(device)

    # print(summary(model, input_size=[2]))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(train_loader):
            # if gpu
            # samples = samples.to(device)
            # labels = labels.to(device)

            outputs = model(samples)
            loss = criterion(outputs, labels)

            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f"epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps},"
                      f"loss = {loss.item():.4f}")
