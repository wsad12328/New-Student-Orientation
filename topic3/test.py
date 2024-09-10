# import torch
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# # 假設我們有三個序列，它們的長度分別是 4, 2, 1
# sequences = [torch.tensor([1, 2, 3, 4]), 
#              torch.tensor([5, 6]), 
#              torch.tensor([7])]

# # 填充序列，使它們的長度相同（4 是最大長度）
# padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
# # 填充結果：
# # tensor([[1, 2, 3, 4],
# #         [5, 6, 0, 0],
# #         [7, 0, 0, 0]])

# # 定義每個序列的實際長度
# lengths = torch.tensor([4, 2, 1])

# # 打包序列
# packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)

# print(packed_sequences)


# padded, lengths = pad_packed_sequence(packed_sequences, batch_first=True)

# print(padded)
# print(lengths)

import torch
import torch.nn as nn

# 假設我們有一個 batch_size 為 3，序列長度為 5，特徵數為 10 的輸入
batch_size = 3
seq_len = 5
input_size = 10
hidden_size = 20
n_layers = 5

# 創建一個 LSTM
lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)

# 假設輸入數據
x = torch.randn(batch_size, seq_len, input_size)

# 初始化隱藏狀態和細胞狀態
h0 = torch.zeros(n_layers, batch_size, hidden_size)
c0 = torch.zeros(n_layers, batch_size, hidden_size)

# 前向傳播，獲得輸出和隱藏狀態
output, (hn, cn) = lstm(x, (h0, c0))

# output: 包含每個時間步的輸出
# hn: 包含最後時間步的隱藏狀態（每層的隱藏狀態）
# cn: 包含最後時間步的細胞狀態（每層的細胞狀態）

# 提取最後一層的隱藏狀態
last_hidden = hn[-1]
print(hn.shape)
print(f"Last hidden state shape: {last_hidden.shape}")
print(last_hidden)
