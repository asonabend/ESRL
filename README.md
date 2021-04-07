# Expert-Supervised Reinforcement Learning (ESRL)

Code for Expert-Supervised Reinforcement Learning (ESRL). If you use our code please cite our [Expert-Supervised Reinforcement Learning for Offline Policy Learning and Evaluation paper](https://arxiv.org/abs/2006.13189).

Repo is set up for Riverswim environment and will work for any episodic, discrete state and action space environment. 

### Overview

Running `main.py` will 

1) Train an expert behavior policy function with PSRL if it's not already present
2) Generate a training dataset using epsilon-greedy behavior policy
3) Train an ESRL policy
4) Evaluate the policy online 
5) Use offline policy evaluation using step-importance sampling (IS), step-weighted importance sampling (WIS) and model-based ESRL to obtain reward estimates

Results are saved in a dictionary or added into the existing results dictionary.

### Function:

To begin the process with defaults run:
```
python main.py
```
The following argument options are available:
```
python main.py --seed 0 --episodes 300 --risk_aversion .1 --epsilon .1 --MDP_samples_train 250 --MDP_samples_eval 500
```
see the [ESRL paper](https://arxiv.org/abs/2006.13189) or our [ESRL video](https://www.youtube.com/watch?v=2f9h1kjfdCM&t=12s) for the details on the arguments and method.

### Bibtex

```
@inproceedings{ASW2020expertsupervised,
 author = {Sonabend, Aaron and Lu, Junwei and Celi, Leo Anthony and Cai, Tianxi and Szolovits, Peter},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {18967--18977},
 publisher = {Curran Associates, Inc.},
 title = {Expert-Supervised Reinforcement Learning for Offline Policy Learning and Evaluation},
 url = {https://proceedings.neurips.cc/paper/2020/file/daf642455364613e2120c636b5a1f9c7-Paper.pdf},
 volume = {33},
 year = {2020}
}

```
