import redis
from collections import namedtuple

from jericho import load_bindings, FrotzEnv
from jericho.util import clean
from jericho.defines import TemplateAction
from jericho.template_action_generator import TemplateActionGenerator

Observation = namedtuple('Observation', ('game', 'look', 'inv'))


def get_tkn_idx_dictionary(env):
    vocab_idx2tkn = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab_idx2tkn[0] = ' '
    vocab_idx2tkn[1] = '<s>'
    vocab_tkn2idx = {v: i for i, v in vocab_idx2tkn.items()}
    return vocab_idx2tkn, vocab_tkn2idx


class JerichoEnv:
    ''' Returns valid actions at each step of the game. '''
    def __init__(self, rom_path, seed, redis_port, step_limit=None):
        self.seed = seed
        self.rom_path = rom_path
        self.bindings = load_bindings(rom_path)
        self.act_gen = TemplateActionGenerator(self.bindings)

        self.steps = 0
        self.step_limit = step_limit
        self.redis_port = redis_port

        self.env = None
        self.tmpl_idx2str = None
        self.obj_idx2str = None
        self.obj_str2idx = None
        self.conn = None

    def create(self):
        self.env = FrotzEnv(self.rom_path, self.seed)
        self.tmpl_idx2str = self.act_gen.templates
        self.obj_idx2str, self.obj_str2idx = get_tkn_idx_dictionary(self.env)
        self.conn = redis.Redis(host='localhost', port=self.redis_port, db=0)

    def _get_admissible_actions(self, obs):
        ''' Queries Redis for a list of valid actions from the current state. '''
        objs = [o[0] for o in self.env.identify_interactive_objects(obs)]
        obj_ids = [self.obj_str2idx[o[:self.bindings['max_word_length']]] for o in objs]
        world_state_hash = self.env.get_world_state_hash()
        valid_action = self.conn.get(world_state_hash)
        if valid_action is None:
            possible_acts = self.act_gen.generate_template_actions(objs, obj_ids)
            valid_action = self.env.find_valid_actions(possible_acts)
            redis_valid_value = '/'.join([str(a) for a in valid_action])
            self.conn.set(world_state_hash, redis_valid_value)
        else:
            try:
                valid_action = [eval(a.strip()) for a in valid_action.decode('cp1252').split('/')]
            except Exception as e:
                print("Exception: {}. Admissible: {}".format(e, valid_action))

        tmpl_idx = [i for i, _tmpl in enumerate(self.tmpl_idx2str) if _tmpl == 'exit']
        tmpl_idx = len(self.tmpl_idx2str) - 2 if len(tmpl_idx) == 0 else tmpl_idx[0]
        valid_action_if_empty = [TemplateAction(self.tmpl_idx2str[tmpl_idx], tmpl_idx, [])]
        valid_action = valid_action if valid_action is not None else valid_action_if_empty
        valid_action = valid_action if len(valid_action) > 0 else valid_action_if_empty
        return valid_action

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done is True:
            info['look'] = 'unknown'
            info['inv'] = 'unknown'
            tmpl_idx = [i for i, _tmpl in enumerate(self.tmpl_idx2str) if _tmpl == 'exit']
            tmpl_idx = len(self.tmpl_idx2str) - 2 if len(tmpl_idx) == 0 else tmpl_idx[0]
            valid_action_if_empty = [TemplateAction(self.tmpl_idx2str[tmpl_idx], tmpl_idx, [])]
            info['valid_action'] = valid_action_if_empty
            info['if_valid'] = False
        else:
            try:
                save = self.env.save_str()
                info['if_valid'] = self.env.world_changed()  # or done

                # look
                look, _, _, _ = self.env.step('look')
                info['look'] = look
                self.env.load_str(save)

                # inv
                inv, _, _, _ = self.env.step('inventory')
                info['inv'] = inv
                self.env.load_str(save)

                # valid_action
                valid_action = self._get_admissible_actions(obs)
                info['valid_action'] = valid_action

            except Exception as e:
                print('Exception occurred in env.py')
                print('{}: {}, Done: {}, Info: {}'.format(e, clean(obs), done, info))

                info['look'] = 'unknown'
                info['inv'] = 'unknown'
                tmpl_idx = [i for i, _tmpl in enumerate(self.tmpl_idx2str) if _tmpl == 'exit']
                tmpl_idx = len(self.tmpl_idx2str) - 2 if len(tmpl_idx) == 0 else tmpl_idx[0]
                valid_action_if_empty = [TemplateAction(self.tmpl_idx2str[tmpl_idx], tmpl_idx, [])]
                info['valid_action'] = valid_action_if_empty
                info['if_valid'] = False

        # step
        self.steps += 1

        # done
        if (self.step_limit is not None) and (self.steps >= self.step_limit):
            done, reward = True, 0
            if self.steps > self.step_limit:
                obs, info = self.reset()

        info['step'] = self.steps

        return obs, reward, done, info

    def reset(self):
        # reset
        obs, info = self.env.reset()
        save = self.env.save_str()

        # step
        self.steps = 0

        # look
        look, _, _, _ = self.env.step('look')
        info['look'] = look
        self.env.load_str(save)

        # inv
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv
        self.env.load_str(save)

        # valid_action
        valid_action = self._get_admissible_actions(obs)
        info['valid_action'] = valid_action

        info['if_valid'] = False
        info['step'] = self.steps

        return obs, info

    def get_dictionary(self):
        if not self.env:
            self.create()
        return self.env.get_dictionary()

    def get_action_set(self):
        return None

    def close(self):
        self.env.close()
