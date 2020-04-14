# Goal-Reaching DQN

## Documents

You may find in this repository all the files that are necessary for completing the coursework:

- ```Task.pdf``` containing all the main coursework instructions and questions.
- ```starter_code.py``` providing Python 3 code which you will build upon during this tutorial and the associated coursework.
- ```environment.py``` in which the environment is implemented. **This file should not be modified**.
- ```torch_example.py``` which gives an example of a supervised learning experiment in PyTorch (see section 2 in ```Tutorial.pdf``` for more information).

## Requirements

You need to use Python 3.6 or greater.

## Installing the environment on a Unix system 

To install the libraries, start by cloning this repository and enter the created folder:

```shell script
git clone https://github.com/alvaroprat97/RL.git
```

Virtual environment (called ```venv``` here):

```shell script
python3 -m venv ./venv 
```

Enter the environment:
```shell script
source venv/bin/activate
```

And install the libraries in the environment by launching the following command:
```shell script
pip install -r requirements.txt
```

This will install the following libraries (and their dependencies) in the virtual environment ```venv```:

- ```torch``` 
- ```opencv-python```
- ```numpy```
- ```matplotlib```

## How to run a script ?

Before launching your experiment, be sure to use the right virtual environment in your shell:
```shell script
source venv/bin/activate  # To launch in the project directory
```

Once you are in the right virtual environment, you can directly launch the scripts 
by using one of the following command:
```shell script
python torch_example.py  # To launch the pytorch example script
python starter_code.py  # To launch the coursework script
```

It is also possible to use the virtual environment tools already included in IDEs (such as PyCharm).

## Leaving the virtual environment

If you want to leave the virtual environment, you just need to enter the following command:
```shell script
deactivate
```
