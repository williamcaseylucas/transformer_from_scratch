import torch
import torch.nn as nn

"""
Flow:
(Each Q, K, V has its own weights and biases)
Q -> Query is represtation of embedding used to determine how much attention a token embedding should should have
K -> Key is representation of the same or different embedding
V -> Yet another representation of the same or different embedding

1. Q@K.T: The dot product between the Query and the Key
2. Run this through a softmax function to see how much attention each token should have
3. Take softmax output and multiply by value to get the final attention output
4. To preserve original continuity of embeddings, concatenate the attention outputs with them
"""


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "embed_size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # key is source sentence
        # query_len is from target (ground truth)

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # queries: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # values: (N, value_len, heads, head_dim)
        # energy: (N, heads, query_len, key_len)

        # sums along the last dimension (d)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            # If element on mask is 0, set energy to -1e20 (-infinity), which after running through Softmax, will be 0
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # if key_len is source sentence and query_len is target sentence, and if softmax is 0.8, we are giving 80% weight to the first word in the source sentence
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)

        # attention: (N, heads, query_len, key_len)
        # values: (N, value_len, heads, head_dim)
        # note: key_len == value_len
        # we want -> N, query_len, heads, head_dim and then flatten -> N, query_len, heads * head_dim
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # note we use key and value
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        # Multi-head attention -> Add & Normalize -> Feed Forward + Add & Normalize
        # Batch norm -> takes average for every batch
        # Layer norm -> takes average for every layer (example in a dimension) -> more computation
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # expand embed size, then pass it back down to a smaller dimension embed size
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # attention + query because of skip connection
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,  # For input embeddings
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,  # how long max sentence is
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # To put all the values in this
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape  # batch_size x sentence_length

        # expand works by essentially repeating the embedding N times
        # 0, 1, 2, 3 -> N times
        # Positional encoding values are used to distinguish each element from others
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # pass in out for q, v, k
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)

        # For the other 2/3
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # src_mask is to ignore padding values and not do unnecessary computation

        # First part of decoder
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))

        # from encoder and query from decoder
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        # Second dimension has to match
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            # enc_out for value and key but we create query
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="mps",
        max_length=100,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # set to 1 if not equal to src_pad_idx
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # Set to 1 if not equal to trg_pad_idx
        trg_mask = torch.tril(
            torch.ones((trg_len, trg_len)).expand(N, 1, trg_len, trg_len)
        ).to(self.device)
        # (N, 1, trg_len, trg_len)

        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)

        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
