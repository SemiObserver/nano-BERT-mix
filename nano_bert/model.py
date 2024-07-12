import math

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module): # for BERT and ALBERT
    def __init__(self, n_embed, max_seq_len):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model) with positional encodings
        pe = torch.zeros(max_seq_len, n_embed)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Div term represents the frequency of the sine and cosine functions
        div_term = torch.exp(torch.arange(0, n_embed, 2).float() * (-torch.log(torch.tensor(10000.0)) / n_embed))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension for broadcasting
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register pe as a buffer, which means it's not a parameter but should be part of the state
        self.register_buffer('pe', pe)
        

        self.layer_norm = torch.nn.LayerNorm(n_embed, eps=1e-12, elementwise_affine=True) # eps: added as sqrt(var + eps) to prevent zero denominator
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False) # inplace=False: do not replace the input by dropouted input
    
    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + self.pe[:x.size(0), :]
        embeddings = self.layer_norm(x)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class BertEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, n_embed): # n_embed = 3, max_seq_len = 16
        super().__init__()

        self.word_embeddings = torch.nn.Embedding(vocab_size, n_embed) # number of words is length of text, each words has length n_embed

    def forward(self, x):
        words_embeddings = self.word_embeddings(x)

        return words_embeddings
    
class ALBertEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, n_embed, n_hid = 3): # n_embed = 3
        super().__init__()

        self.hid_embeddings = torch.nn.Embedding(vocab_size, n_hid) # number of words is length of text, each words has length n_embed

        self.word_embeddings = torch.nn.Embedding(n_hid, n_embed) # number of words is length of text, each words has length n_embed
    def forward(self, x):
        hid_embeddings = self.hid_embeddings(x)
        
        words_embeddings = self.word_embeddings(hid_embeddings)

        return words_embeddings

class BertAttentionHead(torch.nn.Module):
    """
    A single attention head in MultiHeaded Self Attention layer.
    The idea is identical to the original paper ("Attention is all you need"),
    however instead of implementing multiple heads to be evaluated in parallel we matrix multiplication,
    separated in a distinct class for easier and clearer interpretability
    """

    def __init__(self, head_size, dropout, n_embed): # dropout = 0.1, n_embed = 3
        super().__init__()

        self.query = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.key = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.values = torch.nn.Linear(in_features=n_embed, out_features=head_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        # B, Seq_len, N_embed
        B, seq_len, n_embed = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.values(x)

        weights = (q @ k.transpose(-2, -1)) / math.sqrt(n_embed)  # (B, Seq_len, Seq_len)
        weights = weights.masked_fill(mask == 0, -1e9)  # mask out not attended tokens

        scores = F.softmax(weights, dim=-1)
        scores = self.dropout(scores)

        context = scores @ v

        return context


class BertSelfAttention(torch.nn.Module):
    """
    MultiHeaded Self-Attention mechanism as described in "Attention is all you need"
    """

    def __init__(self, n_heads, dropout, n_embed): # , n_heads = 1, dropout = 0.1, n_embed = 3
        super().__init__()

        head_size = n_embed // n_heads

        n_heads = n_heads

        self.heads = torch.nn.ModuleList([BertAttentionHead(head_size, dropout, n_embed) for _ in range(n_heads)])

        self.proj = torch.nn.Linear(head_size * n_heads, n_embed)  # project from multiple heads to the single space

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        context = torch.cat([head(x, mask) for head in self.heads], dim=-1)

        proj = self.proj(context)

        out = self.dropout(proj)

        return out


class FeedForward(torch.nn.Module):
    def __init__(self, dropout, n_embed): # dropout=0.1, n_embed=3
        super().__init__()

        self.ffwd = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.GELU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.ffwd(x)

        return out


class BertLayer(torch.nn.Module):
    """
    Single layer of BERT transformer model
    """

    def __init__(self, n_heads, dropout, n_embed): # n_heads=1, dropout=0.1, n_embed=3
        super().__init__()

        # unlike in the original paper, today in transformers it is more common to apply layer norm before other layers
        # this idea is borrowed from Andrej Karpathy's series on transformers implementation
        self.layer_norm1 = torch.nn.LayerNorm(n_embed)
        self.self_attention = BertSelfAttention(n_heads, dropout, n_embed)

        self.layer_norm2 = torch.nn.LayerNorm(n_embed)
        self.feed_forward = FeedForward(dropout, n_embed)

    def forward(self, x, mask):
        x = self.layer_norm1(x)
        x = x + self.self_attention(x, mask)

        x = self.layer_norm2(x)
        out = x + self.feed_forward(x)

        return out


class BertEncoder(torch.nn.Module):
    def __init__(self, n_layers, n_heads, dropout, n_embed): # n_layers=2, n_heads=1, dropout=0.1, n_embed=4
        super().__init__()

        self.layers = torch.nn.ModuleList([BertLayer(n_heads, dropout, n_embed) for _ in range(n_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class ALBertEncoder(torch.nn.Module):
    def __init__(self, n_layers, n_heads, dropout, n_embed): # n_layers=2, n_heads=1, dropout=0.1, n_embed=3
        super().__init__()
        self.n_layers = n_layers
        self.layer = BertLayer(n_heads, dropout, n_embed)

    def forward(self, x, mask):
        for i in range(n_layers):
            x = layer(x, mask)
        return x

class BertPooler(torch.nn.Module):
    def __init__(self, dropout, n_embed): # dropout=0.1, n_embed=3
        super().__init__()

        self.dense = torch.nn.Linear(in_features=n_embed, out_features=n_embed)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        pooled = self.dense(x)
        out = self.activation(pooled)

        return out


class NanoBERT(torch.nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, dropout, n_embed, max_seq_len): # n_layers=2, n_heads=1, dropout=0.1, n_embed=4, max_seq_len = 16
        super().__init__()
        self.embedding = BertEmbeddings(vocab_size, n_embed)
#         self.embedding = ALBertEmbeddings(vocab_size, n_embed)
#         self.embedding = DEBertAEmbeddings(vocab_size, max_seq_len, n_embed-1, 1)
        self.position = PositionalEncoding(n_embed, max_seq_len)
        self.encoder = BertEncoder(n_layers, n_heads, dropout, n_embed)
        self.pooler = BertPooler(dropout, n_embed)

    def forward(self, x):
        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = (x != 1).unsqueeze(1).repeat(1, x.size(1), 1)
        embeddings = self.embedding(x) 
        position_embeddings = self.position(embeddings)
        encoded = self.encoder(position_embeddings, mask)
        pooled = self.pooler(encoded)
        return pooled

    
class BertMix3(torch.nn.Module):
    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=4, max_seq_len=128, n_classes=2): # n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16, n_classes=2
        super().__init__()
        self.nano_bert = NanoBERT(vocab_size, n_layers, n_heads, dropout, n_embed, max_seq_len)

        self.cls= torch.nn.Linear(in_features=n_embed, out_features=n_classes)
        self.mlm = torch.nn.Linear(in_features=n_embed, out_features=vocab_size)
        self.nsp = torch.nn.Linear(in_features=n_embed, out_features=n_classes)

    def forward(self, input_ids):
        embeddings = self.nano_bert(input_ids)

        r_cls = self.cls(embeddings)
        r_mlm = self.mlm(embeddings)
        r_nsp = self.nsp(embeddings)
        return r_cls, r_mlm, r_nsp

class NanoALBERT(torch.nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, dropout, n_embed, max_seq_len): # n_layers=2, n_heads=1, dropout=0.1, n_embed=4, max_seq_len = 16
        super().__init__()
#         self.embedding = BertEmbeddings(vocab_size, n_embed)
        self.embedding = ALBertEmbeddings(vocab_size, n_embed)
#         self.embedding = DEBertAEmbeddings(vocab_size, max_seq_len, n_embed-1, 1)
        self.position = PositionalEncoding(n_embed, max_seq_len)
        self.encoder = ALBertEncoder(n_layers, n_heads, dropout, n_embed)
        self.pooler = BertPooler(dropout, n_embed)

    def forward(self, x):
        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = (x != 1).unsqueeze(1).repeat(1, x.size(1), 1)
        embeddings = self.embedding(x) 
        position_embeddings = self.position(embeddings)
        encoded = self.encoder(position_embeddings, mask)
        pooled = self.pooler(encoded)
        return pooled
    
class ALBertMix3(torch.nn.Module):
    """
    This is a wrapper on the base NanoBERT that is used for classification task
    One can use this as an example of how to extend and apply nano-BERT to similar custom tasks
    This layer simply adds one additional dense layer for classification
    """

    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=4, max_seq_len=128, n_classes=2): # n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16, n_classes=2
        super().__init__()
        self.nano_bert = NanoALBERT(vocab_size, n_layers, n_heads, dropout, n_embed, max_seq_len)

        self.classifier = torch.nn.Linear(in_features=n_embed, out_features=n_classes)
        self.mlm = torch.nn.Linear(in_features=n_embed, out_features=vocab_size)
        self.nsp = torch.nn.Linear(in_features=n_embed, out_features=n_classes)

    def forward(self, input_ids):
        embeddings = self.nano_bert(input_ids)

        r_cls = self.classifier(embeddings)
        r_mlm = self.mlm(embeddings)
        r_nsp = self.nsp(embeddings)
        return r_cls, r_mlm, r_nsp
####################################################################################################
class AbsolutePositionEmbedding(torch.nn.Module): # for DEBERTA
    def __init__(self, n_embed, max_seq_len):
        super(AbsolutePositionEmbedding, self).__init__()
        
        self.max_seq_len = max_seq_len
        
        self.position_embeddings = nn.Embedding(max_seq_len, n_embed)
        
    def forward(self, input_ids):
        # Get the position IDs from the input IDs
        position_ids = torch.arange(0, self.max_seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get the position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        
        return position_embeddings
    
class DEBertAEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_embed_word = 3, n_embed_pos = 1): 
        super().__init__()

        self.word_embeddings = torch.nn.Embedding(vocab_size, n_embed_word) # number of words is length of text, each words has length n_embed

        self.abposit_embeddings = AbsolutePositionEmbedding(n_embed_pos, max_seq_len)
    def forward(self, x):
        
        words_embeddings = self.word_embeddings(x)
        
        abposits_embeddings = self.abposit_embeddings(x)

        return torch.cat((words_embeddings, abposits_embeddings), dim = 2)

class DEBertAAttentionHead(torch.nn.Module):

    def __init__(self, head_size, dropout, n_embed): # dropout = 0.1, n_embed = 3
        super().__init__()

        self.query = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.key = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.values = torch.nn.Linear(in_features=n_embed, out_features=head_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, I, x, mask):
        B, seq_len, n_embed = x.shape

        q = self.query(I)
        k = self.key(x)
        v = self.values(x)

        weights = (q @ k.transpose(-2, -1)) / math.sqrt(n_embed)  # (B, Seq_len, Seq_len)
        weights = weights.masked_fill(mask == 0, -1e9)  # mask out not attended tokens

        scores = F.softmax(weights, dim=-1)
        scores = self.dropout(scores)

        context = scores @ v

        return context


class DEBertASelfAttention(torch.nn.Module):
    """
    MultiHeaded Self-Attention mechanism as described in "Attention is all you need"
    """

    def __init__(self, n_heads, dropout, n_embed): # , n_heads = 1, dropout = 0.1, n_embed = 3
        super().__init__()

        head_size = n_embed // n_heads

        n_heads = n_heads

        self.heads = torch.nn.ModuleList([DEBertAAttentionHead(head_size, dropout, n_embed) for _ in range(n_heads)])

        self.proj = torch.nn.Linear(head_size * n_heads, n_embed)  # project from multiple heads to the single space

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, I, x, mask):
        context = torch.cat([head(I, x, mask) for head in self.heads], dim=-1)

        proj = self.proj(context)

        out = self.dropout(proj)

        return out
class DEBertALayer(torch.nn.Module):
    """
    First layer of DEBERTA transformer model
    """

    def __init__(self, n_heads, dropout, n_embed): # n_heads=1, dropout=0.1, n_embed=3
        super().__init__()

        # unlike in the original paper, today in transformers it is more common to apply layer norm before other layers
        # this idea is borrowed from Andrej Karpathy's series on transformers implementation
        self.layer_norm1 = torch.nn.LayerNorm(n_embed)
        self.self_attention = DEBertASelfAttention(n_heads, dropout, n_embed)

        self.layer_norm2 = torch.nn.LayerNorm(n_embed)
        self.feed_forward = FeedForward(dropout, n_embed)

    def forward(self, I, x, mask):
        x = self.layer_norm1(x)
        x = x + self.self_attention(I, x, mask)

        x = self.layer_norm2(x)
        out = x + self.feed_forward(x)

        return out
    
class DEBertAEncoder(torch.nn.Module):
    def __init__(self, n_layers, n_heads, dropout, n_embed): # n_layers=2, n_heads=1, dropout=0.1, n_embed=4
        super().__init__()

        self.layers = torch.nn.ModuleList([DEBertALayer(n_heads, dropout, n_embed) for _ in range(n_layers)])

    def forward(self, I, x, mask):
        i = 0
        for layer in self.layers:
            if i == 0:
                x = layer(I, x, mask)
            else:
                x = layer(x, x, mask)
        return x
    
class NanoDEBERTA(torch.nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, dropout, n_embed, max_seq_len): # n_layers=2, n_heads=1, dropout=0.1, n_embed=4, max_seq_len = 16
        super().__init__()
#         self.embedding = BertEmbeddings(vocab_size, n_embed)
#         self.embedding = ALBertEmbeddings(vocab_size, n_embed)
        self.embedding = DEBertAEmbeddings(vocab_size, max_seq_len, n_embed-1, 1)
        self.embedding_I = AbsolutePositionEmbedding(n_embed, max_seq_len)
        self.encoder = DEBertAEncoder(n_layers, n_heads, dropout, n_embed)
        self.pooler = BertPooler(dropout, n_embed)

    def forward(self, x):
        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = (x != 1).unsqueeze(1).repeat(1, x.size(1), 1)
        embeddings = self.embedding(x) 
        embeddings_I = self.embedding_I(x)
        encoded = self.encoder(embeddings_I, embeddings, mask)
        pooled = self.pooler(encoded)
        return pooled
    
class DEBertAMix3(torch.nn.Module):
    """
    This is a wrapper on the base NanoBERT that is used for classification task
    One can use this as an example of how to extend and apply nano-BERT to similar custom tasks
    This layer simply adds one additional dense layer for classification
    """

    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=4, max_seq_len=128, n_classes=2): # n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16, n_classes=2
        super().__init__()
        self.nano_bert = NanoDEBERTA(vocab_size, n_layers, n_heads, dropout, n_embed, max_seq_len)

        self.classifier = torch.nn.Linear(in_features=n_embed, out_features=n_classes)
        self.mlm = torch.nn.Linear(in_features=n_embed, out_features=vocab_size)
        self.nsp = torch.nn.Linear(in_features=n_embed, out_features=n_classes)

    def forward(self, input_ids):
        embeddings = self.nano_bert(input_ids)

        r_cls = self.classifier(embeddings)
        r_mlm = self.mlm(embeddings)
        r_nsp = self.nsp(embeddings)
        return r_cls, r_mlm, r_nsp