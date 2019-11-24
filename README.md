# mdp-assignment


1. Setup Mini Conda:

Download Conda:
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

Install conda:
sh Miniconda3-latest-Linux-x86_64.sh

2. Create  virtual env (Use the environment.yml):
conda env create --file environment.yml

3. Activate environment:
conda activate ml-raghu

4. Run the following commands:
pip install gym
pip install pymdptoolbox



5. Run the experiments using: This will run all the experiments for both the MDP problems:
PYTHONPATH=../:. python -W ignore  Experiment.py
PYTHONPATH=../:. python -W ignore  ExperimentForest.py



References:
1. https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

2. https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb

3. https://medium.com/@anirbans17/reinforcement-learning-for-taxi-v2-edd7c5b76869

4. https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

5.https://github.com/sawcordwell/pymdptoolbox/blob/master/src/mdptoolbox/mdp.py

6. https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa