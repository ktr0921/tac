import copy
import random
import numpy as np

import torch
import torch.nn.functional as F

from models.layer import SharedTextEncoderNetwork, ScoreAdditionStateNetwork, \
    ValueNetwork, ActorNetwork, QNetwork, SequentialTemplateDecoderNetwork, SequentialObjectDecoderNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TAC(torch.nn.Module):
    def __init__(self, sp, tmpl_idx2str, obj_idx2str, embedding_dim, hidden_dim,
                 v_act_expl_prob, exp_v_act_expl_prob, exp_v_act_expl_prob_a, exp_v_act_expl_prob_max,
                 v_act_trn, max_len=None, tau=0.001):
        super(TAC, self).__init__()

        self.sp, self.tmpl_idx2str, self.obj_idx2str, self.tau = sp, tmpl_idx2str, obj_idx2str, tau
        sp_size, self.tmpl_size, self.obj_size = len(sp), len(tmpl_idx2str), len(obj_idx2str)
        sp_embedding_dim, encoded_dim = embedding_dim, hidden_dim
        self.v_act_expl_prob = v_act_expl_prob
        self.exp_v_act_expl_prob = exp_v_act_expl_prob
        self.exp_v_act_expl_prob_a = exp_v_act_expl_prob_a
        self.exp_v_act_expl_prob_max = exp_v_act_expl_prob_max
        self.v_act_trn = v_act_trn
        self.scr2prob = None

        # enc
        self.text_encoder_network = SharedTextEncoderNetwork(sp_size, sp_embedding_dim, hidden_dim, max_len)
        self.state_network = ScoreAdditionStateNetwork(embedding_dim, encoded_dim, hidden_dim)

        # agt
        self.value_network = ValueNetwork(hidden_dim)
        self.actor_network = ActorNetwork(hidden_dim, encoded_dim)
        self.q_network_1 = QNetwork(hidden_dim, encoded_dim)
        self.q_network_2 = QNetwork(hidden_dim, encoded_dim)

        self.target_value_network = copy.deepcopy(self.value_network)

        # dec
        self.template_decoder_network = SequentialTemplateDecoderNetwork(encoded_dim, hidden_dim, self.tmpl_size)
        self.object_decoder_network = SequentialObjectDecoderNetwork(encoded_dim, hidden_dim, self.obj_size)

    def update_value_network(self):
        params = self.value_network.named_parameters()
        target_params = self.target_value_network.named_parameters()

        value_state_dict = dict(params)
        target_value_state_dict = dict(target_params)

        for name in value_state_dict:
            value_state_dict[name] \
                = self.tau * value_state_dict[name].clone() + (1 - self.tau) * target_value_state_dict[name].clone()

        self.target_value_network.load_state_dict(value_state_dict)

    def get_act_str(self, act_idx):
        return self.tmpl_idx2str[act_idx[0].item()] \
                .replace('OBJ', self.obj_idx2str[act_idx[1].item()], 1) \
                .replace('OBJ', self.obj_idx2str[act_idx[2].item()], 1)

    def sample_act_from_dist(self, dist, mask, sample):
        ''' sample template/object from dist '''
        softmax_dist = F.softmax(dist.squeeze(0), dim=-1)
        softmax_dist_copy = softmax_dist.detach().clone()
        if mask is not None:
            softmax_dist_copy[mask] = -torch.tensor(1e-5).to(device)

        selected_idx = (softmax_dist_copy + torch.tensor(1e-5)).multinomial(num_samples=1) if sample is True \
            else (softmax_dist_copy + torch.tensor(1e-5)).topk(1)[1]

        return softmax_dist, selected_idx

    def encode_act(self, act_str):
        act_sp = [self.sp.EncodeAsIds(act_i.replace('OBJ', ' ')) for act_i in act_str]
        encoded_action = self.text_encoder_network.forward_a(act_sp)
        return encoded_action

    def forward_enc_n_agt(self, obs_sp, score):
        game, look, inv = self.text_encoder_network.forward_o(obs_sp)     # text_encoder_network
        state = self.state_network(game, look, inv, score)                # state_network
        state_value = self.value_network(state)                           # value_network
        action = self.actor_network(state)                                # actor_network

        return state, state_value, action

    def get_mask(self, v_act_idx, idx, size):
        if v_act_idx is None:
            return None

        act_mask = torch.ones(len(v_act_idx), size, dtype=torch.bool).to(device)
        for env, v_act_idx_env in enumerate(v_act_idx):
            act_v_env = [va[idx] for va in v_act_idx_env if va[idx] != -1]
            if len(act_v_env) == 0:
                act_mask[env] = 0
            else:
                act_mask[env, torch.tensor(act_v_env).to(device)] = 0

        return act_mask

    def filter_v_act(self, v_act_idx, idx, sel_idx):
        if v_act_idx is None:
            return None

        v_act_idx_out = []
        for sel_idx_env, v_act_idx_env in zip(sel_idx, v_act_idx):
            v_act_idx_out_env = [va for va in v_act_idx_env if (va[idx] == sel_idx_env.item()) or (va[idx] == -1)]
            v_act_idx_out.append(v_act_idx_out_env)

        return v_act_idx_out

    def forward_dec_n_q(self, state, action, v_act_idx=None, trg_act_idx=None, sample=True):
        ''' get the output from the actor network for selected action index (update) '''
        trg_tmpl_idx, trg_obj1_idx, trg_obj2_idx = (None, None, None) if trg_act_idx is None \
            else trg_act_idx.transpose(0, 1).unsqueeze(-1)
        v_act_idx = v_act_idx if self.v_act_trn is True else None

        # tmpl
        tmpl_dist, h_tmpl   = self.template_decoder_network(action)
        tmpl_mask           = self.get_mask(v_act_idx, 0, self.tmpl_size)
        tmpl_dist, tmpl_idx = self.sample_act_from_dist(tmpl_dist, tmpl_mask, sample) \
            if trg_tmpl_idx is None else (F.softmax(tmpl_dist.squeeze(0), dim=-1), trg_tmpl_idx)

        # enc_act for tmpl
        act_str   = [self.tmpl_idx2str[tmpl_i.item()] for tmpl_i in tmpl_idx.squeeze(-1)]
        enc_act   = self.encode_act(act_str)
        v_act_idx = self.filter_v_act(v_act_idx, 0, tmpl_idx)

        # obj1
        obj1_dist, h_obj1   = self.object_decoder_network(action, enc_act, h_tmpl)
        obj1_mask           = self.get_mask(v_act_idx, 1, self.obj_size)
        obj1_dist, obj1_idx = self.sample_act_from_dist(obj1_dist, obj1_mask, sample) \
            if trg_obj1_idx is None else (F.softmax(obj1_dist.squeeze(0), dim=-1), trg_obj1_idx)

        # enc_act for obj1
        act_str   = [act_str_i.replace('OBJ', self.obj_idx2str[obj1_i.item()], 1)
                     for act_str_i, obj1_i in zip(act_str, obj1_idx.squeeze(-1))]
        enc_act   = self.encode_act(act_str)
        v_act_idx = self.filter_v_act(v_act_idx, 1, obj1_idx)

        # obj2
        obj2_dist, h_obj2 = self.object_decoder_network(action, enc_act, h_obj1)
        obj2_mask = self.get_mask(v_act_idx, 2, self.obj_size)
        obj2_dist, obj2_idx = self.sample_act_from_dist(obj2_dist, obj2_mask, sample) \
            if trg_obj2_idx is None else (F.softmax(obj2_dist.squeeze(0), dim=-1), trg_obj2_idx)

        # enc_act for obj2
        act_str   = [act_str_i.replace('OBJ', self.obj_idx2str[obj2_i.item()], 1)
                     for act_str_i, obj2_i in zip(act_str, obj2_idx.squeeze(-1))]
        enc_act   = self.encode_act(act_str)

        # act_idx
        act_idx = torch.cat((tmpl_idx, obj1_idx, obj2_idx), dim=-1)

        # q_value
        q_value_1 = self.q_network_1(state, enc_act)
        q_value_2 = self.q_network_2(state, enc_act)

        return tmpl_dist, obj1_dist, obj2_dist, act_idx, act_str, q_value_1, q_value_2

    def scr2prob_update(self, all_scr, avg_scr):
        avg_scr = int(avg_scr) + 1
        all_scr = [scr_i for scr_i in all_scr if scr_i <= avg_scr]
        min_scr = 0 if min(all_scr) < 0 else min(all_scr)
        max_scr = max(all_scr)

        a = self.exp_v_act_expl_prob_a * (1 / max_scr) if max_scr > 0 else self.exp_v_act_expl_prob_a

        scr = np.arange(min_scr, max_scr + 1)
        print(f'Score\n{scr}')
        if len(scr) == 1:
            self.scr2prob = None
        else:
            prob = np.exp(a * scr)
            norm = (self.exp_v_act_expl_prob_max - self.v_act_expl_prob) / (prob[-1] - prob[0])
            scale = self.exp_v_act_expl_prob_max - (self.exp_v_act_expl_prob_max - self.v_act_expl_prob) * prob[-1] / (prob[-1] - prob[0])
            prob = prob * norm + scale

            self.scr2prob = {scr_i: prob_i for scr_i, prob_i in zip(scr, prob)}

            print(f'Score => Prob')
            for k, v in self.scr2prob.items():
                print(f'{k} => {v}')

    def forward(self, obs_sp, score, v_act_idx=None, trg_act_idx=None, sample=True):
        ''' sample the action output for training (training) '''
        state, state_value, action = self.forward_enc_n_agt(obs_sp, score)
        tmpl_dist, obj1_dist, obj2_dist, act_idx, act_str, q_value_1, q_value_2 \
            = self.forward_dec_n_q(state, action, v_act_idx=v_act_idx, trg_act_idx=trg_act_idx, sample=sample)

        # get prob
        if (self.exp_v_act_expl_prob is True) and (self.scr2prob is not None):
            prob = [self.scr2prob[scr_i] if scr_i in self.scr2prob.keys() else 1.0 for scr_i in score]
        elif self.exp_v_act_expl_prob is True:
            prob = [1.0 for scr_i in score]
        else:
            prob = [self.v_act_expl_prob for scr_i in score]

        # if v_act_idx is given, this means its training, not testing
        # if trg_act_idx is not given, this means its either training or testing, not update
        if (v_act_idx is not None) and (trg_act_idx is None):
            new_act_idx, new_act_str, v_act_expl, random_prob = [], [], [], random.random()
            for i, (act_idx_i, act_str_i, prob_i) in enumerate(zip(act_idx, act_str, prob)):
                if random_prob < prob_i:
                    new_act_idx_i = torch.tensor(random.sample(v_act_idx[i], k=1)[0]).to(device)
                    new_act_idx_i[new_act_idx_i == -1] = 0
                    new_act_str_i = self.get_act_str(new_act_idx_i)
                    v_act_expl.append(True)
                    new_act_idx.append(new_act_idx_i)
                    new_act_str.append(new_act_str_i)
                else:
                    v_act_expl.append(False)
                    new_act_idx.append(act_idx_i)
                    new_act_str.append(act_str_i)

            act_idx = torch.stack(new_act_idx, dim=0).to(device)
            act_str = new_act_str.copy()
        else:
            v_act_expl = [False] * len(prob)

        # act_idx and act_str are only used for training
        return state, state_value, action, tmpl_dist, obj1_dist, obj2_dist, \
            act_idx, act_str, q_value_1, q_value_2, v_act_expl, prob


#     def init_network(self):
#         self.encoder     = Encoder(self.sp, self.max_len, self.embedding_dim, self.hidden_dim)        # enc
#         self.conditioner = ScoreEmbedder(self.hidden_dim)                                             # cnd
#         self.actor       = Actor(self.encoder, self.hidden_dim, self.tmpl_idx2str, self.obj_idx2str)  # act
#         self.critic      = Critic(self.hidden_dim, self.tau)                                          # crt
#
#     def update_value_network(self):
#         self.critic.update_value_network()
#
#     def forward_tac(self, obs_sp, score, trg_act_idx, sample):
#         # model
#         state  = self.encoder(obs_sp)
#         cnd_id = self.conditioner(score)
#         state  = torch.cat([state, cnd_id], dim=1)
#
#         actor_out = self.actor(state, trg_act_idx, sample)
#         action, tmpl_dist, obj1_dist, obj2_dist, act_idx, act_str, enc_act = actor_out
#
#         critic_out = self.critic(state, enc_act)
#         state_value, q1_value, q2_value = critic_out
#         q_value = torch.cat([q1_value, q2_value], dim=-1).to(device)
#
#         # data
#         data = {
#             'state': state, 'state_value': state_value, 'action': action,
#             'tmpl_dist': tmpl_dist, 'obj1_dist': obj1_dist, 'obj2_dist': obj2_dist,
#             'q_value': q_value,
#         }
#         return act_idx, act_str, data
#
#     def forward(self, obs_sp, score, v_act_idx, sample=True):
#         trg_act_idx, mode = None, 1
#         if (self.test is False) and (random.random() < self.v_act_expl_prob):
#             trg_act_idx = torch.tensor([random.sample(_idx, k=1)[0] for _idx in v_act_idx]).to(device)
#             trg_act_idx[trg_act_idx == -1] = 0
#             mode = 2
#
#         act_idx, act_str, data = self.forward_tac(obs_sp, score, trg_act_idx, sample)
#         data['mode'] = [mode] * len(obs_sp)
#         return act_idx, act_str, data
#
#     def forward_trg_state_value(self, next_obs_sp, next_score):
#         next_state       = self.encoder(next_obs_sp)
#         next_cnd_id      = self.conditioner(next_score)
#         next_state       = torch.cat([next_state, next_cnd_id], dim=1)
#
#         next_state_value = self.critic.forward_trg_state_value(next_state)
#         return next_state_value
#
#     def forward_q(self, state, act_str):
#         enc_act            = self.encoder.forward_act(act_str)
#         q1_value, q2_value = self.critic.forward_q(state, enc_act)
#         return q1_value, q2_value


# class TCA(torch.nn.Module):
#     def __init__(self, *args):
#         super().__init__()
#         self.sp, self.tmpl_idx2str, self.obj_idx2str, self.embedding_dim, self.hidden_dim, \
#             self.max_len, self.v_act_expl_prob, self.enc_dropout = args
#
#         self.encoder  = None  # enc
#         self.reasoner = None  # rsn
#         self.actor    = None  # act
#         self.test = False
#
#     def init_network(self):
#         self.encoder  = Encoder(self.sp, self.max_len, self.embedding_dim, self.hidden_dim, init_size=5)  # enc
#         self.actor    = Actor(self.encoder, self.hidden_dim, self.tmpl_idx2str, self.obj_idx2str)         # act
#         self.reasoner = Reasoner(self.sp, self.encoder, self.actor, 32, 1, self.hidden_dim)               # rsn
#
#     def forward_tca(self, obs_sp, score, done, prev_task_id=None, curr_task_id=None, trg_act_idx=None, sample=True):
#         # model
#         state = self.encoder(obs_sp)
#         tsk_loc_dist, tsk_tmpl_dist, tsk_obj1_dist, tsk_obj2_dist, prev_task_id, p, curr_task_id, curr_task \
#             = self.reasoner(state, score, done, prev_task_id=prev_task_id, curr_task_id=curr_task_id)
#
#         state = torch.cat([state, curr_task], dim=1)
#         action, tmpl_dist, obj1_dist, obj2_dist, act_idx, act_str, enc_act \
#             = self.actor(state, trg_act_idx, sample)
#
#         # data
#         data = {
#             'state': state, 'action': action,
#             'tmpl_dist': tmpl_dist, 'obj1_dist': obj1_dist, 'obj2_dist': obj2_dist,
#             'prev_tsk': prev_task_id, 'cmp_prob': p, 'curr_tsk': curr_task_id,
#             'tsk_loc_dist': tsk_loc_dist, 'tsk_tmpl_dist': tsk_tmpl_dist,
#             'tsk_obj1_dist': tsk_obj1_dist, 'tsk_obj2_dist': tsk_obj2_dist,
#             'mode': [9] * len(obs_sp),
#         }
#         return act_idx, act_str, data
#
#     def forward_update(self, obs_sp, score, done, prev_task_id, curr_task_id, trg_act_idx, sample=True):
#         # model
#         state     = self.encoder(obs_sp)
#         p, tsk    = self.reasoner.predict_cmp_prob_and_task(state, prev_task_id, score)
#         curr_task = self.reasoner.encode_task(curr_task_id)
#         state     = torch.cat([state, curr_task], dim=1)
#         action, tmpl_dist, obj1_dist, obj2_dist, act_idx, act_str, enc_act \
#             = self.actor(state, trg_act_idx, sample)
#
#         tsk_update, curr_task_id_list = [], []
#         for i in range(len(curr_task_id)-1):
#             if curr_task_id[i] != curr_task_id[i+1]:
#                 tsk_update.append(tsk[i])
#                 curr_task_id_list.append(curr_task_id[i])
#         tsk_update.append(tsk[-1])
#         curr_task_id_list.append(curr_task_id[-1])
#         tsk_update = torch.stack(tsk_update, dim=0)
#
#         tsk_loc_dist, tsk_tmpl_dist, tsk_obj1_dist, tsk_obj2_dist, tsk_act_idx, tsk_act_str, tsk_loc \
#             = self.reasoner.decode_task(tsk_update, curr_task_id_list)
#
#         # data
#         data = {
#             'state': state, 'action': action,
#             'tmpl_dist': tmpl_dist, 'obj1_dist': obj1_dist, 'obj2_dist': obj2_dist,
#             'prev_tsk': prev_task_id, 'cmp_prob': p, 'curr_tsk': curr_task_id,
#             'tsk_loc_dist': tsk_loc_dist, 'tsk_tmpl_dist': tsk_tmpl_dist,
#             'tsk_obj1_dist': tsk_obj1_dist, 'tsk_obj2_dist': tsk_obj2_dist,
#             'curr_tsk_update': curr_task_id_list,
#             'mode': [9] * len(obs_sp),
#         }
#         return act_idx, act_str, data
#
#     def forward(self, obs_sp, score, done, sample=True):
#         act_idx, act_str, data = self.forward_tca(obs_sp, score, done, sample=sample)
#         return act_idx, act_str, data
