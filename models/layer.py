import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.env import Observation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


class SharedTextEncoderNetwork(torch.nn.Module):
    def __init__(self, sp_size, sp_embedding_dim, hidden_dim, max_len=9999, dropout=0.0, ln=False):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(sp_size, sp_embedding_dim)
        self.embedding_sa = nn.Embedding(4, hidden_dim)
        self.encoder = nn.GRU(sp_embedding_dim, hidden_dim)

        self.ln = nn.LayerNorm(sp_embedding_dim) if ln is True else None
        self.dropout = nn.Dropout(p=dropout)

    def packed_rnn(self, x, rnn, h):
        """ Runs the provided rnn on the input x. Takes care of packing/unpacking. """
        h_pad = torch.tensor(h).to(device).repeat((1, len(x)))
        h_emb = self.embedding_sa(h_pad)

        # x = [x_i[:self.max_len] for x_i in x]
        x = tuple([n if len(n) > 0 else [1, 0, 2] for n in x])  # sanity check: <s> <unk> </s>
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
        x_tt = x_tt.index_select(0, idx_sort)

        if (x_tt[x_tt >= self.embedding.num_embeddings].nelement() > 0) or (x_tt[x_tt < 0].nelement() > 0):
            print('error: x is out of index')
            print('=== x')
            print(x)
            print('=== x_tt')
            print(x_tt)
            x_tt[x_tt >= self.embedding.num_embeddings] = 0
            x_tt[x_tt < 0] = 0
            print('=== FIXED x_tt')
            print(x_tt)

        # Run the embedding layer
        embed = self.embedding(x_tt).permute(1, 0, 2)  # Time x Batch x EncDim
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
        out, _ = rnn(packed, h_emb)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)

        if self.ln is not None:
            out = self.ln(out)
        out = self.dropout(out)

        return out

    def forward_a(self, act_sp):
        act = self.packed_rnn(act_sp, self.encoder, 0)

        return act  # [num_envs=8, txt_embedding_dim=128]

    # [num_envs=8, (game, look, inv)]
    def forward_o(self, obs_sp):
        obs_sp = Observation(*zip(*obs_sp))

        game = self.packed_rnn(obs_sp.game, self.encoder, 1)
        look = self.packed_rnn(obs_sp.look, self.encoder, 2)
        inv  = self.packed_rnn(obs_sp.inv, self.encoder, 3)

        return game, look, inv  # [num_envs=8, txt_embedding_dim=128]


class ScoreAdditionStateNetwork(torch.nn.Module):
    def __init__(self, embedding_dim, encoded_dim, hidden_dim, dropout=0.0, ln=False):
        super(ScoreAdditionStateNetwork, self).__init__()
        self.embedding_score = nn.Embedding(1024, hidden_dim)

        self.tf  = nn.Linear((3 * encoded_dim), hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.s   = nn.Linear(hidden_dim, hidden_dim)

        self.ln = nn.LayerNorm(hidden_dim) if ln is True else None
        self.dropout = nn.Dropout(p=dropout)

    def score_to_embedding(self, score):
        score_embedding = [1024 + score_i if score_i < 0 else score_i for score_i in score]
        score_embedding = torch.tensor(score_embedding).to(device)
        score_embedding = self.embedding_score(score_embedding)
        return score_embedding

    # [num_envs=8, hidden_dim=128], [8, 128], [8, 128], [8, score_dim=10]
    def forward(self, game, look, inv, score):
        h0 = torch.cat([game, look, inv], dim=1)
        h0 = F.elu(self.tf(h0))
        h0 = self.dropout(h0)

        score = self.score_to_embedding(score)
        h0 = h0 + score

        h1 = F.elu(self.fc1(h0))
        h1 = self.dropout(h1)
        h2 = F.elu(self.fc2(h1))
        h2 = self.dropout(h2)
        h3 = F.elu(self.fc3(h2))
        h3 = self.dropout(h3)
        s  = F.elu(self.s(h3))
        s  = self.dropout(s)

        if self.ln is not None:
            s = self.ln(s)

        if torch.isnan(s).sum().item():
            print('=== s has nan')
            print(s)
            quit()
            # s = torch.nan_to_num(s)
            # print('=== changed to 0')
            # print(s)

        if torch.isinf(s).sum().item():
            print('=== s has inf')
            print(s)
            quit()

        return s  # [num_envs=8, hidden_dim=128]


class ValueNetwork(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.v   = nn.Linear(hidden_dim, 1)

    # [num_envs=8, hidden_dim=128]
    def forward(self, s):
        h1 = F.elu(self.fc1(s))
        h2 = F.elu(self.fc2(h1))
        h3 = F.elu(self.fc3(h2))
        v  =        self.v(h3)

        return v  # [num_envs=8, 1]


class ActorNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, encoded_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.a   = nn.Linear(hidden_dim, encoded_dim)

    # [num_envs=8, hidden_dim=128]
    def forward(self, s):
        h1 = F.elu(self.fc1(s))
        h2 = F.elu(self.fc2(h1))
        h3 = F.elu(self.fc3(h2))
        a  = F.elu(self.a(h3))

        if torch.isnan(a).sum().item():
            print('=== a has nan')
            print(a)
            # a = torch.nan_to_num(a)
            # print('=== changed to 0')
            # print(a)

        if torch.isinf(a).sum().item():
            print('=== a has inf')
            print(a)

        return a  # [num_envs=8, encoded_dim=128]


class QNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, encoded_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_dim + encoded_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q   = nn.Linear(hidden_dim, 1)

    # [num_envs=8, hidden_dim=128], [num_envs=8, embedding_dim=128]
    def forward(self, s, a):
        h0 = torch.cat((s, a), dim=-1)
        h1 = F.elu(self.fc1(h0))
        h2 = F.elu(self.fc2(h1))
        h3 = F.elu(self.fc3(h2))
        q  =        self.q(h3)

        return q  # [num_envs=8, 1]


class SequentialTemplateDecoderNetwork(torch.nn.Module):
    def __init__(self, encoded_dim, hidden_dim, tmpl_size, dropout=0.0, ln=(False, False)):
        super(SequentialTemplateDecoderNetwork, self).__init__()
        self.hidden_dim = hidden_dim

        self.tmpl_gru = nn.GRU(encoded_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.tmpl = nn.Linear(hidden_dim, tmpl_size)

    # [num_envs=8, embedding_dim=128]
    def forward(self, a):
        h_init = self.init_hidden(a.size(0))
        h0 = a.unsqueeze(0)

        h1, h_tmpl = self.tmpl_gru(h0, h_init)
        h2         = F.elu(self.fc2(h1))

        tmpl_dist  = self.tmpl(h2)

        if torch.isnan(tmpl_dist).sum().item():
            print('=== tmpl_dist has nan')
            print(tmpl_dist)

        if torch.isinf(tmpl_dist).sum().item():
            print('=== tmpl_dist has inf')
            print(tmpl_dist)

        return tmpl_dist, h_tmpl

    def init_hidden(self, batch=1):
        return torch.zeros(1, batch, self.hidden_dim).to(device)


class SequentialObjectDecoderNetwork(torch.nn.Module):
    def __init__(self, encoded_dim, hidden_dim, obj_size, dropout=0.0, ln=(False, False)):
        super(SequentialObjectDecoderNetwork, self).__init__()

        self.obj_gru = nn.GRU(2 * encoded_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.obj = nn.Linear(hidden_dim, obj_size)

    def forward(self, a, encoded_a, h_in):
        a = a.unsqueeze(0)
        encoded_a = encoded_a.unsqueeze(0)
        h0 = torch.cat((a, encoded_a), dim=-1)

        h1, h_obj = self.obj_gru(h0, h_in)
        h2        = F.elu(self.fc2(h1))

        obj_dist  = self.obj(h2)

        if torch.isnan(obj_dist).sum().item():
            print('=== obj_dist has nan')
            print(obj_dist)

        if torch.isinf(obj_dist).sum().item():
            print('=== obj_dist has inf')
            print(obj_dist)

        return obj_dist, h_obj
