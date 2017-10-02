# Machine Learning Engineer Nanodegree
# Reinforcement Learning
## Project: Train a Smartcab How to Drive

### Install

This project requires **Python 2.7** with the [pygame](https://www.pygame.org/wiki/GettingStarted
) library installed

### Code

Template code is provided in the `smartcab/agent.py` python file. Additional supporting python code can be found in `smartcab/enviroment.py`, `smartcab/planner.py`, and `smartcab/simulator.py`. Supporting images for the graphical user interface can be found in the `images` folder. While some code has already been implemented to get you started, you will need to implement additional functionality for the `LearningAgent` class in `agent.py` when requested to successfully complete the project. 

### Run

In a terminal or command window, navigate to the top-level project directory `smartcab/` (that contains this README) and run one of the following commands:

```python smartcab/agent.py```  
```python -m smartcab.agent```
or
```runsmart -c 1 > logs\logs-c1.txt```
```runsmart -c 2 > logs\logs-c2.txt```
```runsmart -c 3 > logs\logs-c3.txt```
```runsmart -c 9 > logs\logs-c9.txt```

c= 1 : if running "no learning"  (for question 2)
c= 2 : if running "default learning"  (for question 5)
c= 3 : if running "improved learning" (for question 6)
c= 4 : running 1 -> 2 -> 3 in sequencial order

This will run the `agent.py` file and execute your agent code.
