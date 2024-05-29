import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset

from model.Net1D import Net1D


class MyDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        self.max_segments = self._find_max_segments()

    def _find_max_segments(self):
        max_len = max(len(sample) for sample in self.data)
        max_segments = (max_len + self.seq_length - 1) // self.seq_length
        return max_segments

    def __getitem__(self, index):
        sample = self.data[index]
        segments = []
        mask = []

        for start in range(0, len(sample), self.seq_length):
            end = start + self.seq_length
            segment = sample[start:end]
            if len(segment) < self.seq_length:
                segment = np.pad(segment, (0, self.seq_length - len(segment)), 'constant', constant_values=0)
            mask.append(1)
            segments.append(segment)

        while len(segments) < self.max_segments:
            segments.append(np.zeros(self.seq_length))
            mask.append(0)

        segments_array = np.array(segments)
        segments_array = torch.tensor(segments_array, dtype=torch.float).unsqueeze(-1)

        return segments_array, torch.tensor(mask, dtype=torch.float)

    def __len__(self):
        return len(self.data)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super(TemporalAttention, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.swish = Swish()
        self.linear2 = nn.Linear(input_dim, 1, bias=False)
        self.bilinear_weight = nn.Parameter(torch.Tensor(input_dim, input_dim))

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.bilinear_weight)

        self.attention_weights = None

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, input_dim)
        """
        x = self.linear1(x)  # (batch_size, seq_len, input_dim)
        x = self.swish(x)
        x = torch.matmul(x, self.bilinear_weight)
        x = self.linear2(x)  # (batch_size, seq_len, 1)
        self.attention_weights = F.softmax(x, dim=1)
        return self.attention_weights


class DeepSpiro(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Parameters:
        n_classes: number of classes

    """

    def __init__(self, in_channels, out_channels, n_len_seg, n_classes, device, verbose=False):
        super(DeepSpiro, self).__init__()

        self.n_segments = None
        self.n_samples = None
        self.n_seg = None
        self.n_length = None
        self.n_channel = None
        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.device = device
        self.verbose = verbose

        # (batch, channels, length)
        self.cnn = Net1D(
            in_channels=1,
            base_filters=8,
            ratio=1.0,
            filter_list=[16, 32, 32, 64],
            m_blocks_list=[2, 2, 2, 2],
            kernel_size=16,
            stride=2,
            groups_width=1,
            verbose=False,
        )
        self.out_channels = self.cnn.filter_list[-1]

        # (batch, seq, feature)
        self.rnn = nn.LSTM(input_size=self.out_channels,
                           hidden_size=self.out_channels,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        self.temporal_attention = TemporalAttention(input_dim=2 * self.out_channels)
        self.dense = nn.Linear(2 * self.out_channels, n_classes)

    def forward(self, x, mask):
        self.n_samples, self.n_segments, self.n_len_seg, self.n_channel = x.shape
        lengths = mask.sum(dim=1).tolist()
        # (n_samples, n_segments, n_len_seg, n_channel)
        out = x
        # (n_samples, n_segments, n_len_seg, n_channel) -> (n_samples*n_segments, n_len_seg, n_channel)
        out = out[mask > 0]
        if self.verbose:
            print(out.shape)
        # (n_samples*n_seg, n_len_seg, n_channel) -> (n_samples*n_seg, n_channel, n_len_seg)
        out = out.permute(0, 2, 1)
        # out = out.permute(1, 0, 2)
        if self.verbose:
            print(out.shape)
        # cnn
        out = self.cnn(out)
        if self.verbose:
            print(out.shape)
        # global avg, (n_samples*n_seg, out_channels)
        out = out.mean(-1)
        if self.verbose:
            print(out.shape)
        combined_out = torch.zeros(self.n_samples, self.n_segments, self.out_channels, device=out.device)
        valid_segment_idx = 0
        for i in range(self.n_samples):
            num_valid_segments = int(mask[i].sum().item())
            valid_segments = out[valid_segment_idx: valid_segment_idx + num_valid_segments]
            combined_out[i, mask[i].bool(), :] = valid_segments
            valid_segment_idx += num_valid_segments
        if self.verbose:
            print("After Reconstructing:", combined_out.shape)
        out = combined_out.view(-1, self.n_segments, self.out_channels)
        if self.verbose:
            print(out.shape)
        out = rnn_utils.pack_padded_sequence(combined_out, lengths, batch_first=True, enforce_sorted=False)
        # rnn
        packed_output, _ = self.rnn(out)
        out, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        attention_weights = self.temporal_attention(out)
        out = torch.sum(out * attention_weights, dim=1)
        out = self.dense(out)
        if self.verbose:
            print(out.shape)

        return out
