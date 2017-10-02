##
## This is prepared for Udacity Machine Learning Engineer Nanodegree online class
## Author: jtmoogle @github.com All Rights Reserved
## Date: Aug/Sept, 2017
## 
## This file /smartcab/agent.py was mainly for smart cab agent.
## Note: 
## Q(s, a) is the Q value of the (s, a) pair
## R(s) is the immediate reward to enter the state 's'.
## Q(s', a')  s' is the state that you will be in after you take action 'a' in state 's'. There is a transition matrix. In our case, the transition is deterministic i.e the state s' can be predicted with probability 1. In real life that is not true.
## a' - is the action that gives us the max value of Q in state s'.
##

import numpy.random as random
import math
import numpy as np
import sys, getopt
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

fyi=True
default_Qvalue = 0.0
default_epsilonvalue = 0.0
default_decayvalue = 0.05
default_alphavalue = 0.0
default_seed = 168

def zstr(s): 
    """ if s contains None, it replaced with 'noraffice' string
    """
    return s or "notraffic"
        

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        random.seed( default_seed )
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        
        ###########
        # Set any additional class parameters as needed
        self.ntrial = 0
        self.verbose = self.env.verbose
        self.default_decay = False
    
    def use_default_decay(self, usedefault=True): 
        """ The use_default_decay function is called at the beginning of each trial.
            'usedefault' is set to True, default decay function 
            will be e + 1 = e - 0.05 for trial number t """

        self.default_decay = usedefault
            
            
        
    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        self.ntrial += 1
        

        if testing:  # In environment.py logic if total trial > 20 or learning or epasilon < tolerance 
            self.epsilon = default_epsilonvalue
            self.alpha   = default_alphavalue
        else:
            # decay function
            if (self.default_decay):   
                # for question 6 default learning, if default learning, 
                # use default decay function e t+1 = e - 0.05 for trial number t
                self.epsilon -= default_decayvalue  # e-0.05  default learning CA 
            else: 
                # for 0 < alpha < 1  tried out various functions (1)  ϵ=a**t  (2)  ϵ=e(−at)   (3) ϵ=cos(at)  
                #    track  safetly & reliability  for <optized=False>  <optimzed=True>
                if self.alpha > 0.0 and self.alpha < 1.0:
            #    self.epsilon -= 1.0 / 20000 * self.ntrial  # default learning improved learning: A+A+ FB
            # x  self.epsilon = math.exp( -1.0 * self.alpha * self.ntrial )  # e(−at) FF FF
            # v  self.epsilon = math.exp( -0.01 * self.alpha * self.ntrial ) # e(−at) CA+ A+A
            # vv self.epsilon = math.exp( -0.0125 * self.alpha * self.ntrial ) # e(−at)  (1) DA+ A+A+  (2) A+A+A+A+ 
            #    self.epsilon = math.exp( -0.5 * self.alpha * self.ntrial )    # e(−at) FF FF
            #    self.epsilon = math.exp( -0.1 * self.alpha * self.ntrial )    # e(−at) FD FF
            #    self.epsilon = math.exp( -0.05 * self.alpha * self.ntrial )   # e(−at) FD FA
            #    self.epsilon = math.exp( -0.025 * self.alpha * self.ntrial )  # e(−at) FA A+A
            # v  self.epsilon = math.exp( -0.0125 * self.alpha * self.ntrial ) # e(−at) DA+ A+
            # x  self.epsilon = math.fabs(math.cos( self.alpha * self.ntrial)) # |cos(at)| (1) FC FA+  (2) FB FB
            # x  self.epsilon = math.fabs( math.cos(self.alpha * self.ntrial)) # |cos(at)| FC FF
            # x  self.epsilon = math.cos(self.alpha * self.ntrial)    #FF FF
                    self.epsilon = math.fabs( math.cos(0.0125 * self.alpha * self.ntrial)) # |cos(at)| (1) A+C A+A+  log_5 (2) BD+A+A+ log log8
            #    self.epsilon = math.fabs( math.cos(0.02 * self.alpha * self.ntrial))   # |cos(at)| (1) FA FA (2) FA+A+A+ log10
            #     self.epsilon = math.fabs( math.cos(0.125 * self.alpha * self.ntrial)) # |cos(at)| FD FF
            #    self.epsilon = math.fabs( math.cos(0.05 * self.alpha * self.ntrial))   # |cos(at)| FC FA
            #    self.epsilon = math.fabs( math.cos(0.025 * self.alpha * self.ntrial))  # |cos(at)| FA+ CA
            #vv   self.epsilon = math.fabs( math.cos(0.01 * self.alpha * self.ntrial))  # |cos(at)| (1) FA A+A+  (2) A+A+A+A+ log7 9
            # x  self.epsilon = math.pow(self.alpha, self.ntrial)            # a**t  FF FF
            #    self.epsilon = math.pow( 0.025*self.alpha, self.ntrial)     # a**t  FF FF
            # x  self.epsilon = math.pow( 0.00125 *self.alpha, self.ntrial)  # a**t  FF FF
            # x  self.epsilon = math.pow (-0.00125 * self.alpha, self.ntrial)# a**t  FF FF
            # x  self.epsilon = 0.000125 * math.pow(self.alpha, self.ntrial) # a**t  FD FF
            # x  self.epsilon = 0.00125 * math.pow(self.alpha, self.ntrial)  # a**t  FD FF
            #    self.epsilon = 1.025 * math.pow(self.alpha, self.ntrial)    # a**t  FF FF
                else:
                    # if not 0 < alpha < 1   tried out various functions (1) ϵt+1=ϵt−0.05   (2) ϵ=1/t**2              
                    # tolerance - epsilon tolerance before beginning testing, default is 0.05
                    self.epsilon -= default_decayvalue  # e-0.05  (1) FD FF  log_3  (2) A+A+ A+A+ log_6  (3) A+A+ A+A+ log_7 9(4) BD+A+A+ log log8 (5) A+A+A+A+ log10
            # x   self.epsilon = 0.0125 * 1.0/math.pow(self.ntrial, 2)  #  1/t**2 FF FF
            # x   self.epsilon = 1.0 / math.pow(self.ntrial, 2)         # 1/t**2 FF FC
            # x   self.epsilon = 125 * 1.0/math.pow(self.ntrial, 2)     # 125/t**2 FD FD
            # x   self.epsilon = 1.0 /math.pow(self.ntrial, 2)          # 1/t**2 FF FF
            # x   self.epsilon = 1.0 /math.pow(default_seed * self.ntrial, 2)  # 1/(t*default_seed) ** 2 FF FF
                                 
        return None
    

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, 
        # and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent
        # valid inputs from environment.py
        # Tried out vairous features to track safety and reliability
        
                #  (1) Tried out 5 features (waypoint light lef tright  oncoming)  FF FA A+A+
                #  state = ( 'Go-' + waypoint, 'is-'+ inputs['light'] + '-light', 'left-' + zstr(inputs['left']), \
                #                  "right-" + zstr( inputs['right']), 'oncoming-' + zstr(inputs['oncoming']) )
        
        #  (2) Tried out 4 features (light left oncoming waypoint)   FF A+A+ A+A
        #state = ( 'is-'+ inputs['light'] + '-light', 'leftcar-' + zstr(inputs['left']), 
        #         'oncomingcar-' + zstr(inputs['oncoming']), 'nextwaypt-' + waypoint)
        state = ( inputs['light'], zstr(inputs['left']), zstr(inputs['oncoming']), waypoint)
        
                #  (3) Tried out 3 features waypoint light left   FF DF A+B
                #        state = ( 'Go-' + waypoint, 'is-'+ inputs['light'] + '-light', 'left-' + zstr(inputs['left']) )

        if self.verbose: 
            print('-- build_state --> state("light-state", "left-traffic", ' \
                ' "oncoming-traffic","next-waypoint")= {}'.format( state ))
        
        self.createQ(state)
        
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        # Calculate the maximum Q-value of all actions for a given state
        maxQ = max( self.Q.get(state).values() )   
        # maxQ = np.argmax( self.Q.get(state).values() )
        
        return maxQ 



    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        #if verbose: print('-- createQ -->  state=({})   self.Q.get={}'.format(state, self.Q.get(state)))
        #if self.Q.get(state) is None:
            
        if (self.learning) and (not self.Q.has_key(state)):
            self.Q[state] = { 
                    action:default_Qvalue 
                    for action in self.valid_actions 
                    }
            if self.verbose: print('-- CreateQ self learning, not has key --> self.Q[{}]={}'.format(state, self.Q[state]))
        return 



    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None
        
        ########### 
        ## select a policy
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        
        if (not self.learning):  # no learning, choose random action w
            action = random.choice(self.valid_actions)
        elif random.random() < self.epsilon:
            # learning &  random < epsilon, 
            # choose random action with epsilon probability
            action = random.choice(self.valid_actions)
        else :
            # learning &  random > epsilon
            # choose max Q-values of all actions based on current state
            maxQ = self.get_maxQ(state)
            candidates_actions = [k for k,v in self.Q[state].items() if (v == maxQ) ]      
            print( '------maxQ={}-----candidates_actions={}'.format(maxQ, candidates_actions ))
            action = random.choice(candidates_actions)         
        
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')        
        if self.learning:
            # Updated Q-value = current Q-value * (1-alpha) + alpha * (current reward + discount factor * expected future reward)
            current_Qvalue = self.Q[self.state][action] 
            self.Q[self.state][action] = current_Qvalue * (1 - self.alpha) + self.alpha * reward

        return 


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward  
        self.learn(state, action, reward)   # Q-learn
        
        return 


def run(cmd):
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. 
    input: cmd command
        1 : if running 'no learning'   (for question 2)
        2 : if running 'default learning'  (for question 5)
        3 : if running 'improved learning' (for question 6)
        9 : running 1 -> 2 -> 3 in sequencial order
    """
    
    print('> > > - - - Run cmd={}  start - - - > > > '.format(cmd)) 
    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    #env = Environment(verbose=True, num_dummies=5, grid_size=(4,2))
    env = Environment(verbose=True)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    if cmd == 1 :  # run no-learning
        agent = env.create_agent( LearningAgent, learning=False )
    elif cmd == 2 :  # run default-learning
        agent = env.create_agent( LearningAgent, learning=True )
        agent.use_default_decay()  # use default decay function e t+1 = e - 0.05      
    else:     # 3 # run improved learning
        #agent = env.create_agent( LearningAgent, learning=True, epsilon=0.6, alpha=0.4 )
        agent = env.create_agent( LearningAgent, learning=True, epsilon=0.4, alpha=0.4 )
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    if cmd == 1:
        env.set_primary_agent(agent)
    else:     # 2, 3
        env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create a Simulator
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    if cmd == 1:    # run no-learning
        sim = Simulator(env, update_delay=0.01, log_metrics=True, display=True)
    elif cmd == 2:  # default learning
        sim = Simulator(env, update_delay=0.01, log_metrics=True, display=False)
    else:     # 3 improved learning
        sim = Simulator(env, update_delay=0.01, log_metrics=True, display=False, optimized=True )
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    if cmd == 1:
        sim.run(n_test=10)    
    elif cmd == 2:
        sim.run(n_test=10)    
    else:   #3
        sim.run(n_test=25, tolerance=0.0875 )  # tolerance 0.0875 DA A+A+ log_11 0.875 A+A+A+A+  logs
    
    print('> > > - - - Run End - - - > > > ')

#
# main
# 
if __name__ == '__main__':
    """ 
    Parse command line 
    input: args 
        -c 1 : (default) if running 'no learning'   (see question 2)
        -c 2 : if running 'default learning'  (see question 5)
        -c 3 : if running 'improved learning'  (see question 6)
        -c 9 : running 1, 2, 3
        """ 
    cmd = 1
    try:
        argv = sys.argv[1:]
        opts, args =  getopt.getopt(argv,"c:")
    except getopt.GetoptError:
      print 'python -m smartcab.agency -c <digit 1-3, 9 for all>'
      sys.exit(2)
    print( '-- main-> run {} --> opts={} args={}'.format(argv, opts, args))
    
    for opt, arg in opts:
        if opt in ('-c'): 
            cmd = np.int(arg)
            print( '    --> cmd={} arg={}'.format( cmd, arg))    

    if cmd == 9 :
        #execute 1, 2, 3
        run( 1 )
        run( 2 )
        run( 3 )
    else:
        run( cmd )
    