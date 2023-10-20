# Text-based Actor Critic on Text-based Games

Code for CoNLL2023 paper [A Minimal Approach for Natural Language Action Space in Text-based Games]().

# Quickstart

Install Dependencies:
```bash
sudo apt-get install redis-server
pip install -U spacy
pip install --user jericho==2.4.2
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
```

Train TAC on ZORK1:
```bash
python run.py --redis_path <path_to_your_redis_server> --redis_port <redis_port_number>
```

For example:
```bash
python run.py --redis_path /usr/local/Cellar/redis/6.2.1/bin/redis-server --redis_port 6970
```

Train TAC on other games with different seedings:
```bash
python run.py --redis_path <path_to_your_redis_server> --redis_port <redis_port_number> --env_name <game_name> --seed <seeding_number> --lr <learning_rate>
```

For example:
```bash
python run.py --redis_path /usr/local/Cellar/redis/6.2.1/bin/redis-server --redis_port 6970 --env_name deephome --seed 1234 --lr 0.00001
```

# Acknowledgement

The code is based on [TDQN and DRRN](https://github.com/microsoft/tdqn).

# Citation

```
@inproceedings{dkryu2023conll,
    title = "A Minimal Approach for Natural Language Action Space in Text-based Games",
    author = "Ryu, Dongwon Kelvin and Fang, Meng and Haffari, Gholamreza and Pan, Shirui and Shareghi, Ehsan",
    booktitle = "Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL)",
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```
