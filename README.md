# Autonomous Vehicle Reinforcement Learning using DDPG Algorithm v1.1.2
A CLI tool for running DRL experiments on a simulated Autonomous vehicle platoon (or many).

## Install
To install, create a venv and install the requirements file.
This program is loosely tested on Python 3.6, 3.7 and 3.8
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note: For simulations to render, you will need
gym==0.21.0
pyglet==1.5.23

***Note that on windows you will perform ```venv/Scripts/Activate``` instead of ```source venv/bin/activate```.

## Run the program
To see what the cli can do, enter the following.
```
python run.py --help
```

### Training
```
python run.py tr
```
To gain insight on configurable parameters, use
```
python run.py tr --help
```
A training session will create a .outputs folder within the base of the repo.
This folder will contain an experiment folder containing the results.

### Reporting
After training, you can generate a latex report in a folder. This can be embedded into an overleaf project as a latex subfile if you like.
```
python run.py lmany
```
Note that this command only works if the .outputs folder contains an experiment(s).

### Figure Generation
If you wish to compare similar experiments, you can use at the [accumulator.py](./workers/accumulator.py) file.

```
python run.py accumr
```
This will accumulate experiment results and plot the reward averaged across the seeds in svg files for each platoon.

### Re-run simulations
You can re-run simulations on existing experiments, by running the below command and passing the file as an argument
```
python run.py esim
```

## Notes
You can nest --help on any of the above commands if you cant figure out what to do with the cli.

