# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from util import flipCoin

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

        '''
        qvalues is going to be like this {(state, action): qvalue}
        '''
        self.qvalues = {}




    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if (state, action) in self.qvalues:
            return self.qvalues[(state, action)]
        else: 
            return 0.0


        #util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.

        """
        """
        value of state = action with the highest qvalue

        1 get legal actions from state
        2 loop each action and retrieve qvalue
        3 if getQvalue(state, action)
        4 keep track of max and compare them

        """

        actions = self.getLegalActions(state)

        if not actions: #if empty 
            return 0.0
        
        max_value = float("-inf")

        for i in actions:

            max_value = max(self.getQValue(state, i), max_value)


        return max_value


        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.

        """
        '''
        For every  action
        save action with max value
        if 2 actions tie then random.chocie 
        other return none
        '''

        best_action = []
        actions = self.getLegalActions(state)
        max_value = float("-inf")

        if not actions: #if empty then action = none
            return None;
        
        for i in actions:

            qvalue = self.getQValue(state, i)

            if qvalue > max_value:

                max_value = qvalue
                best_action = [i]

            elif qvalue == max_value:

                best_action.append(i)
                
        #if tie choose randomly against choices 
        answer = random.choice(best_action) 

        return answer
        # util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)


        """
        '''
        basically to decide if we want to be rnadom or pick the best action

        epsilom = probability of agent exploring random actions 
        '''
        # Pick Action
        actions = self.getLegalActions(state)
        agent_pick = None

        if not actions:
            return None

        if flipCoin(self.epsilon):
            #if true then random
 
            agent_pick = random.choice(actions)

        else:
            #if false then best action
            agent_pick = self.computeActionFromQValues(state)

        return agent_pick
        
        #util.raiseNotDefined()


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf

        """
        """
        update qvalues = {}
        
        function
        q(s,a) = (1 - self.alpha) currentQvalue(s, a) + self.alpha[reward + slef.discount * computeValuesfromQvalue(nextState)]
        """

        "*** YOUR CODE HERE ***"

        currentQvalue = self.getQValue(state, action)

        newQvalue = (1 - self.alpha) * currentQvalue + self.alpha * ( reward + (self.discount * self.computeValueFromQValues(nextState)))

        self.qvalues[(state, action)] = newQvalue

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
