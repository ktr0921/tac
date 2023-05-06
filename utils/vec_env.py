import time
import redis
import subprocess
import numpy as np
from multiprocessing import Process, Pipe


def start_redis(redis_path, redis_port):
    print(f'Starting {redis_path} on {redis_port} port')
    subprocess.Popen([redis_path, '--save', '\"\"', '--appendonly', 'no', '--port', str(redis_port)])

    time.sleep(1)


def worker(remote, parent_remote, env):
    parent_remote.close()
    env.create()
    try:
        done = False
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                if done:
                    ob, info = env.reset()
                    reward = 0
                    done = False
                else:
                    ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob, info = env.reset()
                remote.send((ob, info))
            elif cmd == 'close':
                env.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class VecEnv:
    def __init__(self, num_envs, env, redis_path, redis_port):
        start_redis(redis_path, redis_port)
        self.conn = redis.Redis(host='localhost', port=redis_port, db=0)
        self.total_steps = 0

        self.closed = False
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        if self.total_steps % 1024 == 0:
            self.conn.flushdb()
        self.total_steps += 1

        self._assert_not_closed()
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return zip(*results)

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return zip(*results)

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
