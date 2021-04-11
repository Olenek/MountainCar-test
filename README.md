# MountainCar-test
A repo for the testing solutions of Gym-MountainCar problem.

### How to use
Directory `src/` contains Python implementations of different models that may solve the problem (currently q-learning and deep-q-learning methods are implemented).

In directories `models/` and `q-tables/` information about trained agents is stored. 

Jupyter notebook `MountainCar.ipynb` contains examples of usage.

### Performance
Currently, **q-learning agent** is capable of solving the problem in minutes of training (100-500 training episodes), which results in it climbing the mountain in 120-160 steps, depending on the random's leniency and number of training episodes.

**Deep-q-learning** agent has potential of greater training ceiling, although converges at a lower rate (all things considered, probably, its application here is suboptimal).
