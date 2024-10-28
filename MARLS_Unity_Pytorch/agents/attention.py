import torch
import torch.nn as nn

def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = torch.matmul(q, torch.transpose(k, dim0=-2, dim1=-1))  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# ## Multi-head Attention

# In[ ]:


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(input_size, d_model)
        self.wk = nn.Linear(input_size, d_model)
        self.wv = nn.Linear(input_size, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = torch.reshape(x, (-1, x.shape[1], self.num_heads, self.depth))
        return torch.transpose(x, dim0=1, dim1=2)

    def forward(self, v, k, q, mask=None):
        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = torch.transpose(scaled_attention,dim0=1, dim1=2)  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = torch.reshape(scaled_attention,
                                      (-1, scaled_attention.shape[1], self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def main():
    temp_mha = MultiHeadAttention(input_size=60, d_model=256, num_heads=4)
    q = torch.rand((64, 2, 60))  # (batch_size, encoder_sequence, d_model)
    v = torch.rand((64, 10, 60))  # (batch_size, encoder_sequence, d_model)

    out, attn = temp_mha(v, v, q)
    print(out.shape, attn.shape)



if __name__ == '__main__':
    main()