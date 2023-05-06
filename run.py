import os
import argparse

env_name_list = [
    '905', 'acorncourt', 'advent', 'adventureland', 'afflicted', 'anchor', 'awaken', 'balances', 'ballyhoo', 'curses',
    'cutthroat', 'deephome', 'detective', 'dragon', 'enchanter', 'enter', 'gold', 'hhgg', 'hollywood', 'huntdark',
    'infidel', 'inhumane', 'jewel', 'karn', 'lgop', 'library', 'loose', 'lostpig', 'ludicorp', 'lurking', 'moonlit',
    'murdac', 'night', 'omniquest', 'partyfoul', 'pentari', 'planetfall', 'plundered', 'reverb', 'seastalker',
    'sherlock', 'snacktime', 'sorcerer', 'spellbrkr', 'spirit', 'temple', 'theatre', 'trinity', 'tryst205', 'weapon',
    'wishbringer', 'yomomma', 'zenon', 'zork1', 'zork2', 'zork3', 'ztuu'
]


def parse_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument('--env_name', default='zork1', choices=env_name_list)
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--num_envs', default=32, type=int)
    parser.add_argument('--spm_path', default='etc/spm_models/unigram_8k.model')
    parser.add_argument('--rom_path', default=None)
    # output
    parser.add_argument('--output_dir', default='outputs/')
    parser.add_argument('--output_name', default='')
    parser.add_argument('--resume_dir', default=None)
    # redis
    parser.add_argument('--redis_path', default='/usr/local/Cellar/redis/6.2.1/bin/redis-server')
    parser.add_argument('--redis_port', default=6970, type=int)
    # emb
    parser.add_argument('--max_len', default=9999, type=int)
    parser.add_argument('--embedding_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--memory_size', default=100000, type=int)
    # training
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--step', default=100000, type=int)
    # step
    parser.add_argument('--update_step', default=1, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--evaluation_step', default=500, type=int)
    parser.add_argument('--checkpoint_step', default=5000, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--test_episode', default=10, type=int)
    # update
    parser.add_argument('--batch_size', default=64, type=int)
    # --gamma 0.7 did not work. the agent simply tries to achieve 25 only without getting 10 or 5 score.
    # however, this might need more experiments
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)         # 1e-2 ~ 1e-5
    parser.add_argument('--weight_decay', default=1e-6, type=float)  # 5e-2 ~ 1e-5
    parser.add_argument('--clip', default=5, type=float)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # testing
    parser.add_argument('--model', default='tac', choices=['tac'])

    parser.add_argument('--tmpl_sup_loss_coeff',     default=1.0, type=float)  # 3.0; faster learning is better for exploration
    parser.add_argument('--obj_sup_loss_coeff',      default=1.0, type=float)  # 5.0; faster learning is better for exploration
    parser.add_argument('--tmpl_rwd_max_loss_coeff', default=1.0, type=float)
    parser.add_argument('--obj_rwd_max_loss_coeff',  default=1.0, type=float)
    parser.add_argument('--state_value_loss_coeff',  default=1.0, type=float)
    parser.add_argument('--q_loss_coeff',            default=1.0, type=float)

    parser.add_argument('--tau', default=0.001, type=float)
    # per
    parser.add_argument('--experience_replay', default='per', choices=['er', 'per'])
    parser.add_argument('--per_alpha', default=0.7, type=float)  # (prioritization: no 0 - full 1)
    parser.add_argument('--per_beta',  default=0.3, type=float)  # (importance weights: no 0 - full 1)
    # expl
    # --v_act_expl_prob 0.5 & 0.7 did not work. exploitation is problem, the agent finds it difficult to achieve >10 in
    # zork1 during training. 0.2 works good & 0.3 works
    parser.add_argument('--v_act_expl_prob', default=0.30, type=float)
    # dropout
    parser.add_argument('--txt_dropout', default=0.0, type=float)
    parser.add_argument('--stt_dropout', default=0.0, type=float)
    parser.add_argument('--dec_dropout', default=0.0, type=float)
    # layer norm
    parser.add_argument('--txt_ln', action='store_true')
    parser.add_argument('--stt_ln', action='store_true')
    parser.add_argument('--dec_ln', action='store_true')
    # save
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_memory', action='store_true')

    # stability
    parser.add_argument('--advantage_norm', action='store_true')  # do not use. learn too slow

    parser.add_argument('--v_act_trn', action='store_true')

    parser.add_argument('--exp_v_act_expl_prob', action='store_true')
    parser.add_argument('--exp_v_act_expl_prob_a', default=5.0, type=float)
    parser.add_argument('--exp_v_act_expl_prob_max', default=0.7, type=float)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # testing
    parser.set_defaults(advantage_norm=True)

    parser.set_defaults(save_model=True)
    parser.set_defaults(save_memory=True)

    # # state
    # parser.add_argument('--score_binary', action='store_true')
    # parser.add_argument('--score_emb_add', action='store_true')

    # # decoder
    # parser.add_argument('--actor_type', default='sequential', choices=['linear', 'sequential', 'mlm'])

    # parser.add_argument('--weight_init', default=None, choices=['xavier_uniform', 'kaiming'])

    # relu seems to have gradient exploding a lot, so use elu

    return parser.parse_args()


def main():
    args = parse_args()

    if args.rom_path is None:
        _rom_path = [os.path.join('etc/roms', rom) for rom in os.listdir('etc/roms')
                     if args.env_name == rom.rsplit('.', 1)[0]]
        if len(_rom_path) == 1:
            args.rom_path = _rom_path[0]
        else:
            print('rom_path error: too many rom files')
            quit()

    tmp = f'{args.env_name}-s{args.seed}-{args.output_name}'
    args.output_dir = os.path.join(args.output_dir, tmp.strip('-'))
    if os.path.isdir(args.output_dir) is True:
        for i in range(1000):
            if os.path.isdir(args.output_dir + f'-{str(i).zfill(3)}') is False:
                args.output_dir = args.output_dir + f'-{str(i).zfill(3)}'
                break

    print(args)
    if args.model == 'tac':
        from trainers.tac_trainer import TACTrainer
        print(f"=== calling tac ===")
        agent = TACTrainer(args)
        agent.train()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
