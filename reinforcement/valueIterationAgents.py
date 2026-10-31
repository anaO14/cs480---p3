# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    """
    This method updates the value for every state (box in the grid), one round at a time. 
    The agent does not move.
    For each box, it compares the Q-values of the avilable actions and stores the highest one as that box's new value. 
    One round means updating all states using the previous round's values. 

    immediate reward + discount x stored value of next state 
    """
    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #state = self.mdp.getStates()
        #transition = self.mdp.getTransitionStatesAndProbs(state, action)
        #oldValue = self.values[n]
        #immediateReward = transition + self.discount

        newValues = 0

        for iteration in range(self.iterations):
            newValues = util.Counter()

            #loop through all the states 
            for state in self.mdp.getStates():

                #get actions from each states & loop through actions
                actions = self.mdp.getPossibleActions(state)

                if not actions:
                    newValues[state] = 0 
                    continue

                highestQVal = float('-inf')

                #compare q-values for this state's actions 
                for action in actions: 
                    qVal = self.computeQValueFromValues(state, action)

                    #compare returned qVal numbers and save the highest one 
                    if(qVal > highestQVal):
                        highestQVal = qVal

                #save the highest q-value after checking all actions 
                newValues[state] = highestQVal

            #replace old values only after finishing the entire round 
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          probability: how likely an outcome is 
          reward is something you get immediately
          value estimates what you can earn starting from a state over time

          ex: 
          possible next state       probability        stored value 
          box a                    0.8(80% change)          5
          box b                    0.2(20% change)         -2
        """
        "*** YOUR CODE HERE ***"

        qVal = 0

        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):

            #points recieved immediately for a particular transition 
            reward = self.mdp.getReward(state, action, nextState)

            #the value stored for the next possible state 
            nextValue = self.values[nextState]

            #expected total of immediate and discounted future rewards starting from a state 
            qVal += probability * (reward + self.discount * nextValue)

        return qVal
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        highestQVal = float('-inf')
        newValues = util.Counter()
        bestAction = None
        
        actions = self.mdp.getPossibleActions(state)

        if not actions:
            return None

        for action in actions: 
            qVal = self.computeQValueFromValues(state, action)

            #compare returned qVal numbers and save the highest one 
            if(qVal > highestQVal):
                highestQVal = qVal
                bestAction = action

            #save the highest q-value after checking all actions 
            newValues[state] = highestQVal

        return bestAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

