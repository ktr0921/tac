import os
import time
import random
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from jericho.util import clean

from utils import logger
from utils.env import JerichoEnv, Observation
from utils.vec_env import VecEnv
from utils.buffers import ReplayBuffer, PrioritizedReplayBuffer, Transition

from models.tac import TAC

torch.set_printoptions(profile="full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log


class TACTrainer:
    def __init__(self, args):
        print(f"=== initiating under {device} ===")
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            print(f'- current_device: {current_device}')
            print(f'- device_count: {torch.cuda.device_count()}')
            print(f'- get_device_name: {torch.cuda.get_device_name(current_device)}')
            memory_cached = torch.cuda.memory_cached(current_device)
            memory_allocated = torch.cuda.memory_allocated(current_device)
            memory_free = memory_cached - memory_allocated
            print(f'- memory_cached: {memory_cached}')
            print(f'- memory_allocated: {memory_allocated}')
            print(f'- memory_free (memory_cached - memory_allocated): {memory_free}')

        print('=== seed ===')
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        print('=== step ===')
        self.step = args.step
        self.resume_step = 0
        self.log_step = args.log_step
        self.update_step = args.update_step
        self.update_freq = args.update_freq
        self.checkpoint_step = args.checkpoint_step
        self.evaluation_step = args.evaluation_step
        self.tst_scr = set()

        self.num_envs = args.num_envs
        self.per = True if args.experience_replay == 'per' else False
        self.rwd_count = dict()
        self.per_alpha = args.per_alpha
        self.per_beta = args.per_beta

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.clip = args.clip

        self.advantage_norm = args.advantage_norm

        self.tmpl_sup_loss_coeff = args.tmpl_sup_loss_coeff
        self.obj_sup_loss_coeff = args.obj_sup_loss_coeff
        self.tmpl_rwd_max_loss_coeff = args.tmpl_rwd_max_loss_coeff
        self.obj_rwd_max_loss_coeff = args.obj_rwd_max_loss_coeff
        self.state_value_loss_coeff = args.state_value_loss_coeff
        self.q_loss_coeff = args.q_loss_coeff

        self.sp = None

        self.env = None
        self.envs = None

        self.model = None
        self.memory = None

        self.loss_cls = None
        self.loss_rgs = None
        self.optimizer = None

        self.init_trainer(args)

    def init_trainer(self, args):
        print('=== log ===')
        self.output_dir = args.output_dir
        self.resume_dir = args.resume_dir
        configure_logger(self.output_dir)
        os.mkdir(os.path.join(self.output_dir, 'log'))

        print('=== sentence piece ===')  # EncodeAsIds, EncodeAsPieces
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)

        print('=== env ===')
        self.env = JerichoEnv(args.rom_path, args.seed, args.redis_port, args.env_step_limit)
        self.envs = VecEnv(self.num_envs, self.env, args.redis_path, args.redis_port)
        self.env.create()  # Create the environment for evaluation

        print('=== model ===')
        self.model = TAC(
            self.sp, self.env.tmpl_idx2str, self.env.obj_idx2str, args.embedding_dim, args.hidden_dim,
            args.v_act_expl_prob, args.exp_v_act_expl_prob, args.exp_v_act_expl_prob_a, args.exp_v_act_expl_prob_max,
            args.v_act_trn, max_len=args.max_len, tau=args.tau,
        ).to(device)

        print(f'=== model params ===')
        for name, param in self.model.named_parameters():
            name_grad = f'{name} (grad, {param.requires_grad})'
            print(f'{name_grad:60}: {list(param.data.size())}')

        if self.resume_dir is not None:
            self.load()

        if self.per is False:
            print('=== experience replay ===')
            self.memory = ReplayBuffer(args.memory_size)
        else:
            print('=== prioritized experience replay ===')
            self.memory = PrioritizedReplayBuffer(args.memory_size, self.per_alpha)

        print('=== loss ===')
        self.loss_cls = nn.BCELoss()
        self.loss_rgs = nn.SmoothL1Loss()

        print('=== optimizer ===')
        self.optimizer = torch.optim.Adam(
            list(self.model.text_encoder_network.parameters())
            + list(self.model.state_network.parameters())
            + list(self.model.value_network.parameters())
            + list(self.model.actor_network.parameters())
            + list(self.model.q_network_1.parameters())
            + list(self.model.q_network_2.parameters())
            + list(self.model.template_decoder_network.parameters())
            + list(self.model.object_decoder_network.parameters())
            , lr=args.lr, weight_decay=args.weight_decay
        )

        log('arguments:')
        log(args)
        log('===== ===== ===== ===== ===== ===== ===== ===== ===== =====')

    def get_obs_sp(self, obs, info):
        game_sp = [self.sp.EncodeAsIds(obs_i) for obs_i in obs]
        look_sp = [self.sp.EncodeAsIds(info_i['look']) for info_i in info]
        inv_sp = [self.sp.EncodeAsIds(info_i['inv']) for info_i in info]
        return [Observation(_g, _l, _i) for _g, _l, _i in zip(game_sp, look_sp, inv_sp)]

    def get_v_act_idx(self, info):
        return [[[_v.template_id] + _v.obj_ids + [-1] * (2 - len(_v.obj_ids)) for _v in info_i['valid_action']]
                for info_i in info]

    # def get_selected_act_str(self, act_idx):
    #     return [self.env.tmpl_idx2str[_idx[0].item()]
    #                 .replace('OBJ', self.env.obj_idx2str[_idx[1].item()], 1)
    #                 .replace('OBJ', self.env.obj_idx2str[_idx[2].item()], 1)
    #             for _idx in act_idx]

    def train(self):
        # env reset
        obs, info = self.envs.reset()
        obs_sp       = self.get_obs_sp(obs, info)
        v_act_idx    = self.get_v_act_idx(info)
        score        = [info_i['score'] for info_i in info]

        print('==========================')
        print('===== START TRAINING =====')
        print('==========================')
        start = time.time()
        for step in range(self.resume_step + 1, self.step + 1):
            log(f'\n===== Training step {step} =====\n')
            log(f'Game: {clean(obs[0])}')
            log(f'Look: {clean(info[0]["look"])}')
            log(f'Inv: {clean(info[0]["inv"])}')

            # model
            with torch.no_grad():
                state, state_value, action, tmpl_dist, obj1_dist, obj2_dist, act_idx, act_str, \
                    q_value_1, q_value_2, v_act_expl, prob = self.model(obs_sp, score, v_act_idx=v_act_idx, sample=True)

            # env step
            obs, rwd, done, info = self.envs.step(act_str)
            next_obs_sp       = self.get_obs_sp(obs, info)
            next_v_act_idx    = self.get_v_act_idx(info)
            next_score        = [info_i['score'] for info_i in info]
            next_if_valid     = [info_i['if_valid'] for info_i in info]

            log(f'Action: {act_str[0]}')
            log(f'Probability Threshold: {prob[0]}')
            log(f'Admissible Exploration: {v_act_expl[0]}')
            log(f'Reward: {rwd[0]}')
            log(f'Score: {info[0]["score"]}')
            log(f'Done: {done[0]}')

            for rwd_i in rwd:
                if rwd_i in self.rwd_count.keys():
                    self.rwd_count[rwd_i] += 1
                else:
                    self.rwd_count[rwd_i] = 1
            with open(os.path.join(self.output_dir, 'log', 'rwd_cnt.txt'), 'a') as f:
                f.write(f'{self.rwd_count}\n')

            # memory
            memory_idx = []
            for i in range(self.num_envs):
                # tmpl_dist[0, i], obj1_dist[i], obj2_dist[i],
                memory_idx.append(self.memory._next_idx)
                self.memory.add(
                    obs_sp[i], score[i], v_act_idx[i], act_idx[i], next_if_valid[i], rwd[i],
                    next_obs_sp[i], next_score[i], next_v_act_idx[i], done[i]
                )

            if self.per is True:
                q_value = torch.min(q_value_1, q_value_2)
                _, target_q_value = self.get_target(
                    next_obs_sp, next_score,
                    torch.tensor(rwd).to(device), torch.tensor(done).to(device)
                )
                priorities = torch.abs(q_value.squeeze(-1) - target_q_value).cpu().detach().numpy()
                priorities = np.array([priorities_i if priorities_i >= 1e-10 else 1e-10 for priorities_i in priorities])
                self.memory.update_priorities(memory_idx, priorities)

            # next
            obs_sp, score, v_act_idx = next_obs_sp, next_score, next_v_act_idx

            for done_i, info_i in zip(done, info):
                if done_i:
                    tb.logkv_mean('EpisodeScore', info_i['score'])

                    with open(os.path.join(self.output_dir, 'log', 'trn_scr.txt'), 'a') as f:
                        f.write(f'{info_i["score"]},')
            with open(os.path.join(self.output_dir, 'log', 'trn_scr.txt'), 'a') as f:
                f.write(f'\n')

            if step % self.update_step == 0:
                for _ in range(self.update_freq):
                    self.update()

            if step % self.evaluation_step == 0:
                self.evaluate(step, nb_episodes=10, sample=True)
                self.evaluate(step)

            if step % self.checkpoint_step == 0:
                self.save_model(step)
                self.save_memory(step)

            if step % self.log_step == 0:
                tb.logkv('Step', step)
                tb.logkv("Second", round(time.time() - start, 2))
                tb.logkv("StepPerSecond", round((step - self.resume_step) / (time.time() - start), 2))
                tb.dumpkvs()

        self.save_model(self.step)
        self.save_memory(self.step)

        # self.evaluate(self.step, nb_episodes=100, sample=True)
        # self.evaluate(self.step)
        tb.dumpkvs()

        print('=========================')
        print('===== DONE TRAINING =====')
        print('=========================')

    def remove_unused_obj(self, act_idx):
        obj_count = [self.env.tmpl_idx2str[_idx[0]].count('OBJ') for _idx in act_idx]
        removal_matrix = torch.tensor([[1] * (_c + 1) + [0] * (2 - _c) for _c in obj_count]).to(device).transpose(0, 1)
        return removal_matrix

    def update_supervised(self, *args):
        tmpl_dist, obj1_dist, obj2_dist, action, v_act_idx, act_idx = args

        # get the output based on valid action, used to update (supervised) actor networks
        v_tmpl_dist_gt = torch.zeros((self.batch_size, len(self.env.tmpl_idx2str)))
        v_obj1_dist_gt = torch.zeros((self.batch_size, len(self.env.obj_idx2str)))
        v_obj2_dist_gt = torch.zeros((self.batch_size, len(self.env.obj_idx2str)))
        for i in range(self.batch_size):
            for _idx in v_act_idx[i]:
                v_tmpl_dist_gt[i, _idx[0]] = 1
                v_obj1_dist_gt[i, _idx[1]] = 1 if (_idx[1] != -1) and (act_idx[i, 0] == _idx[0]) else 0
                v_obj2_dist_gt[i, _idx[2]] = 1 if (_idx[2] != -1) and (act_idx[i, 0] == _idx[0]) and \
                                                  (act_idx[i, 1] == _idx[1]) else 0

        v_tmpl_dist_gt = v_tmpl_dist_gt.to(device).to(torch.float)
        v_obj1_dist_gt = v_obj1_dist_gt.to(device).to(torch.float)
        v_obj2_dist_gt = v_obj2_dist_gt.to(device).to(torch.float)

        tmpl_sup_loss = self.loss_cls(tmpl_dist, v_tmpl_dist_gt)
        obj1_sup_loss = self.loss_cls(obj1_dist, v_obj1_dist_gt)
        obj2_sup_loss = self.loss_cls(obj2_dist, v_obj2_dist_gt)

        return tmpl_sup_loss, obj1_sup_loss, obj2_sup_loss

    def update_reward_maximization(self, *args):
        weights, tmpl_dist, obj1_dist, obj2_dist, act_idx, q_value, state_value = args
        
        act_idx[act_idx == -1] = 0
        selected_act_ts = act_idx.transpose(0, 1)
        log_prob_tmpl = (tmpl_dist + torch.tensor(1e-5)).gather(1, selected_act_ts[0].unsqueeze(-1)).log()
        log_prob_obj1 = (obj1_dist + torch.tensor(1e-5)).gather(1, selected_act_ts[1].unsqueeze(-1)).log()
        log_prob_obj2 = (obj2_dist + torch.tensor(1e-5)).gather(1, selected_act_ts[2].unsqueeze(-1)).log()

        removal_matrix = self.remove_unused_obj(act_idx)

        advantage = weights * (q_value.squeeze(-1) - state_value.squeeze(-1))
        advantage = advantage.detach()
        advantage = advantage / advantage.abs().max() if self.advantage_norm is True else advantage

        tmpl_rwd_max_loss = -torch.mean(advantage * log_prob_tmpl.squeeze(-1) * removal_matrix[0])
        obj1_rwd_max_loss = -torch.mean(advantage * log_prob_obj1.squeeze(-1) * removal_matrix[1])
        obj2_rwd_max_loss = -torch.mean(advantage * log_prob_obj2.squeeze(-1) * removal_matrix[2])

        return tmpl_rwd_max_loss, obj1_rwd_max_loss, obj2_rwd_max_loss

    def get_target(self, next_obs_sp, next_score, rwd, done):
        with torch.no_grad():
            next_game, next_look, next_inv = self.model.text_encoder_network.forward_o(next_obs_sp)
            next_state = self.model.state_network(next_game, next_look, next_inv, next_score)
            next_state_value = self.model.target_value_network(next_state)
            next_state_value[done] = 0

        target_value = rwd + self.gamma * next_state_value.squeeze(-1)

        return target_value.detach().clone(), target_value.detach().clone()

    def update_value(self, *args):
        state_value, next_obs_sp, next_score, done, rwd, q_value_1, q_value_2 = args
        target_state_value, target_q_value = self.get_target(next_obs_sp, next_score, rwd, done)

        # state_value_loss
        state_value_loss = 0.5 * self.loss_rgs(state_value.squeeze(-1), target_state_value)

        # q_1_loss, q_2_loss
        q_1_loss = 0.5 * self.loss_rgs(q_value_1.squeeze(-1), target_q_value)
        q_2_loss = 0.5 * self.loss_rgs(q_value_2.squeeze(-1), target_q_value)

        return target_q_value, state_value_loss, q_1_loss, q_2_loss

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        if self.per is True:
            transitions = self.memory.sample(self.batch_size, self.per_beta)
            transitions, previous_priorities, weights, idxes = transitions
            weights = torch.tensor(weights).to(device)
        else:
            transitions = self.memory.sample(self.batch_size)
            previous_priorities, idxes = None, None
            weights = torch.tensor(1.0).to(device).to(torch.float)

        # batch
        batch = Transition(*zip(*transitions))
        obs_sp, score, v_act_idx, act_idx, if_valid, rwd, \
            next_obs_sp, next_score, next_v_act_idx, done = batch
        act_idx = torch.stack(act_idx, dim=0).to(device)
        rwd = torch.tensor(rwd).to(device)
        done = torch.tensor(done).to(device)

        # zero grad
        self.optimizer.zero_grad()

        # reconstruct computation graph for buffer
        state, state_value, action, tmpl_dist, obj1_dist, obj2_dist, act_idx, act_str, \
            q_value_1, q_value_2, v_act_expl, prob = self.model(obs_sp, score, trg_act_idx=act_idx)
        q_value = torch.min(q_value_1, q_value_2)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        tmpl_sup_loss, obj1_sup_loss, obj2_sup_loss = self.update_supervised(
            tmpl_dist, obj1_dist, obj2_dist, action, v_act_idx, act_idx
        )
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        tmpl_rwd_max_loss, obj1_rwd_max_loss, obj2_rwd_max_loss = self.update_reward_maximization(
            weights, tmpl_dist, obj1_dist, obj2_dist, act_idx, q_value, state_value
        )
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        target_q_value, state_value_loss, q_1_loss, q_2_loss = self.update_value(
            state_value, next_obs_sp, next_score, done, rwd, q_value_1, q_value_2
        )
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if self.per is True:
            priorities = torch.abs(q_value.squeeze(-1) - target_q_value).cpu().detach().numpy()
            priorities = np.array([priorities_i if priorities_i >= 1e-10 else 1e-10 for priorities_i in priorities])
            self.memory.update_priorities(idxes, priorities)
            # with open(os.path.join(self.output_dir, 'log', 'memory.txt'), 'a') as _f:
            #     f_write = ''
            #     for a, b, c, d, e, f, g, h in sorted(list(zip(act_str, idxes, weights, rwd, previous_priorities, priorities, torch.min(q_value_1, q_value_2), target_q_value)),
            #                              key=lambda x: (x[3], x[4]), reverse=True):
            #         #
            #         f_write += f'{a:25}\t({b:>5} | {round(c.item(), 5):<7} | {d})\t'
            #         f_write += f'[{round(e, 3):<6} => abs({round(g.item(), 3):<6} - {round(h.item(), 3):<6}) * alpha = '
            #         f_write += f'{round((f ** self.per_alpha).item(), 3):>6}]\n'
            #     _f.write(f'{f_write}\n')

        # loss
        loss = self.tmpl_sup_loss_coeff * tmpl_sup_loss \
            + self.obj_sup_loss_coeff * obj1_sup_loss \
            + self.obj_sup_loss_coeff * obj2_sup_loss \
            + self.tmpl_rwd_max_loss_coeff * tmpl_rwd_max_loss \
            + self.obj_rwd_max_loss_coeff * obj1_rwd_max_loss \
            + self.obj_rwd_max_loss_coeff * obj2_rwd_max_loss \
            + self.state_value_loss_coeff * state_value_loss \
            + self.q_loss_coeff * q_1_loss \
            + self.q_loss_coeff * q_2_loss

        loss.backward()

        # step
        unclipped_grad_norm = torch.tensor(0.0).to(device).to(torch.float)
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            unclipped_grad_norm += p.grad.data.norm(2).item()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        clipped_grad_norm = torch.tensor(0.0).to(device).to(torch.float)
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            clipped_grad_norm += p.grad.data.norm(2).item()

        self.optimizer.step()

        tb.logkv_mean('1.1 TmplSupLoss', (self.tmpl_sup_loss_coeff * tmpl_sup_loss).item())
        tb.logkv_mean('1.2 Obj1SupLoss', (self.obj_sup_loss_coeff * obj1_sup_loss).item())
        tb.logkv_mean('1.3 Obj2SupLoss', (self.obj_sup_loss_coeff * obj2_sup_loss).item())
        tb.logkv_mean('2.1 TmplRwdMaxLoss', (self.tmpl_rwd_max_loss_coeff * tmpl_rwd_max_loss).item())
        tb.logkv_mean('2.2 Obj1RwdMaxLoss', (self.obj_rwd_max_loss_coeff * obj1_rwd_max_loss).item())
        tb.logkv_mean('2.3 Obj2RwdMaxLoss', (self.obj_rwd_max_loss_coeff * obj2_rwd_max_loss).item())
        tb.logkv_mean('3.1 StateValueLoss', (self.state_value_loss_coeff * state_value_loss).item())
        tb.logkv_mean('3.2 Q1Loss', (self.q_loss_coeff * q_1_loss).item())
        tb.logkv_mean('3.3 Q2Loss', (self.q_loss_coeff * q_2_loss).item())
        tb.logkv_mean('4.1 Loss', loss.item())
        tb.logkv_mean('4.2 UnclippedGradNorm', unclipped_grad_norm.item())
        tb.logkv_mean('4.3 ClippedGradNorm', clipped_grad_norm.item())

        self.model.update_value_network()

        return loss.item()

    def evaluate(self, training_step, nb_episodes=1, sample=False):
        self.model.eval()
        self.tst_scr = set()
        with torch.no_grad():
            total_score = []
            for ep in range(nb_episodes):
                score = self.evaluate_episode(training_step, ep, sample=sample)
                log(f'===== Evaluation episode {ep} ended with score {score} =====')
                total_score.append(score)

            with open(os.path.join(self.output_dir, 'log', 'tst_scr.txt'), 'a') as f:
                f.write(f'{training_step}\t: ' + ','.join([str(s) for s in total_score]) + '\n')

            avg_score = sum(total_score) / nb_episodes

            self.model.train()
            tb.logkv(f'Sample{sample}EvalScore', avg_score)

            if sample is True:
                self.model.scr2prob_update(self.tst_scr, avg_score)
            return avg_score

    def evaluate_episode(self, training_step, ep, sample=False):
        step, done = 0, [False]
        obs, info = self.env.reset()
        obs, info = [obs], [info]
        mode = 'smp' if sample is True else 'grd'
        self.tst_scr.add(info[0]['score'])

        while not done[0]:
            with open(os.path.join(self.output_dir, 'log', f'tst-{mode}.txt'), 'a') as f:
                f.write(f'=== Evaluation episode {ep} step {step} ===\n')
                f.write(f'Game: {clean(obs[0])}\n')
                f.write(f'Look: {clean(info[0]["look"])}\n')
                f.write(f'Inv: {clean(info[0]["inv"])}\n')

            obs_sp       = self.get_obs_sp(obs, info)
            v_act_idx    = self.get_v_act_idx(info)
            score        = [info_i['score'] for info_i in info]

            with torch.no_grad():
                state, state_value, action, tmpl_dist, obj1_dist, obj2_dist, act_idx, act_str, \
                    q_value_1, q_value_2, v_act_expl, prob = self.model(obs_sp, score, sample=sample)

            selected_act_ts = act_idx.transpose(0, 1)
            selected_act_prob = tmpl_dist.gather(1, selected_act_ts[0].unsqueeze(-1))

            top_20_tmpl = tmpl_dist[0, :].topk(20)[1].tolist()
            top_20_tmpl_str = ' | '.join([
                f'{self.env.tmpl_idx2str[_t]} ({round(tmpl_dist[0, _t].item(), 5)})' for _t in top_20_tmpl
            ])

            valid_q = ''
            for _v in info[0]['valid_action']:
                if (q_value_1 is not None) and (q_value_2 is not None):
                    _v_ids = torch.tensor([[_v.template_id] + _v.obj_ids + [0] * (2 - len(_v.obj_ids))]).to(device)
                    # _act_str = self.get_selected_act_str(_v_ids)
                    _act_str = [self.model.get_act_str(_v_ids_i) for _v_ids_i in _v_ids]
                    _encoded_action = self.model.encode_act(_act_str)
                    q_v_value_1 = self.model.q_network_1(state, _encoded_action)
                    q_v_value_2 = self.model.q_network_2(state, _encoded_action)
                    valid_q += f'{_v.action} ({round(q_v_value_1.item(), 5)}, {round(q_v_value_2.item(), 5)}) | '
                else:
                    valid_q += f'{_v.action} (None, None) | '
            # log
            action_str_log = f'({self.env.tmpl_idx2str[act_idx[0, 0].item()]}, ' \
                             + f'{self.env.obj_idx2str[act_idx[0, 1].item()]}, ' \
                             + f'{self.env.obj_idx2str[act_idx[0, 2].item()]})'
            action_prob_log = f'({[round(n.item(), 5) for n in selected_act_prob[0]]})'

            with open(os.path.join(self.output_dir, 'log', f'tst-{mode}.txt'), 'a') as f:
                f.write(f'State Value: {round(state_value[0].item(), 5)}\n')
                f.write(f'Top 20 Templates: {top_20_tmpl_str}\n')
                f.write(f'Q Value for Valid Action: {valid_q.strip(" |")}\n')
                f.write(f'Action: {action_str_log} => {act_str[0]}\n')
                f.write(f'Probability Threshold: {prob[0]}\n')
                f.write(f'Admissible Exploration: {v_act_expl[0]}\n')
                f.write(f'Action Probability: {act_idx[0]} = {action_str_log} = {action_prob_log}\n')
                q_value = torch.min(q_value_1, q_value_2)
                f.write(f'Q Value: {round(q_value[0].item(), 5)}\n')

            obs, rwd, done, info = self.env.step(act_str[0])
            obs, rwd, done, info = [obs], [rwd], [done], [info]
            step += 1

            self.tst_scr.add(info[0]['score'])

            with open(os.path.join(self.output_dir, 'log', f'tst-{mode}.txt'), 'a') as f:
                f.write(f'Reward: {rwd[0]}\n')
                f.write(f'Score: {info[0]["score"]}\n')
                f.write(f'If Valid: {info[0]["if_valid"]}\n\n')

        return info[0]['score']

    def load(self):
        saved_model = sorted([int(d.strip('model-.pt')) for d in os.listdir(self.resume_dir) if 'model-' in d])
        self.resume_step = saved_model[-1]
        print(f'=== load {os.path.join(self.resume_dir, f"model-{str(self.resume_step).zfill(6)}.pt")} ===')
        self.model = torch.load(os.path.join(self.resume_dir, f'model-{str(self.resume_step).zfill(6)}.pt'), map_location=device)
        print(f'=== load {os.path.join(self.resume_dir, f"memory-{str(self.resume_step).zfill(6)}.pt")} ===')
        self.memory = torch.load(os.path.join(self.resume_dir, f'memory-{str(self.resume_step).zfill(6)}.pt'), map_location=device)

    def save_model(self, number):
        next_number = str(number).zfill(6)
        torch.save(self.model, os.path.join(self.output_dir, f'model-{next_number}.pt'))

        remove_number = sorted([int(d.strip('model-.pt')) for d in os.listdir(self.output_dir)
                                if ('model-' in d) and (int(d.strip('model-.pt')) % 5000 != 0)])
        if len(remove_number) > 5:
            remove_number = str(remove_number[0]).zfill(6)
            os.remove(os.path.join(self.output_dir, f'model-{remove_number}.pt'))

    def save_memory(self, number):
        next_number = str(number).zfill(6)
        torch.save(self.memory, os.path.join(self.output_dir, f'memory-{next_number}.pt'))

        remove_number = sorted([int(d.strip('memory-.pt')) for d in os.listdir(self.output_dir) if 'memory-' in d])
        if len(remove_number) > 1:
            remove_number = str(remove_number[0]).zfill(6)
            os.remove(os.path.join(self.output_dir, f'memory-{remove_number}.pt'))

