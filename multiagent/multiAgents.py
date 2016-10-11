# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from json.encoder import INFINITY
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        curPos = currentGameState.getPacmanPosition()
        foodList = newFood.asList()
        for state in newGhostStates:
            if newPos == state.getPosition():
                return -99999
            elif manhattanDistance(newPos, state.getPosition()) < 2:
                score -= 80
        if len(foodList) > 0:
            closestDist, closestFood = min([(manhattanDistance(curPos, food),food) for food in foodList])    
            if (manhattanDistance(newPos, closestFood)) < (manhattanDistance(curPos, closestFood)):
                score += 20    
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def minimax(self,state,depth,agentIndex=0):
        if state.isWin() or state.isLose() or depth == 0:
            return (self.evaluationFunction(state),)
        numAgents = state.getNumAgents()
        newDepth = depth if agentIndex != (numAgents - 1) else depth - 1
        newAgentIndex = (agentIndex + 1) % numAgents      
        scoreAction = [(self.minimax(state.generateSuccessor(agentIndex,action), newDepth, newAgentIndex)[0], action) 
                 for action in state.getLegalActions(agentIndex)]
        if agentIndex is 0:
            return max(scoreAction)
        else:
            return min(scoreAction)
        
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, self.depth)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBeta(self,state,depth,agentIndex,alpha,beta):
        if state.isWin() or state.isLose() or depth == 0:
            return (self.evaluationFunction(state), )
        numAgents = state.getNumAgents()
        newDepth = depth if agentIndex != (numAgents - 1) else depth - 1
        newAgentIndex = (agentIndex + 1) % numAgents
        actionList = state.getLegalActions(agentIndex)
        if agentIndex is 0:
            v = (-INFINITY, )
            for act in actionList:
                nextState = state.generateSuccessor(agentIndex, act)
                v = max(v, (self.alphaBeta(nextState, newDepth, newAgentIndex, alpha, beta)[0],act))
                if v > beta:
                    return v
                alpha = max(alpha,v)
            return v
        else:
            v = (INFINITY, )
            for act in actionList:
                nextState = state.generateSuccessor(agentIndex, act)
                v = min(v, (self.alphaBeta(nextState, newDepth, newAgentIndex, alpha, beta)[0], act))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBeta(gameState, self.depth, 0,(-(INFINITY), ), (INFINITY, ))[1]
        
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self,state,depth,agentIndex=0):
        if state.isWin() or state.isLose() or depth == 0:
            return (self.evaluationFunction(state),)
        numAgents = state.getNumAgents()
        newDepth = depth if agentIndex != (numAgents - 1) else depth - 1
        newAgentIndex = (agentIndex + 1) % numAgents      
        scoreAction = [(self.expectimax(state.generateSuccessor(agentIndex,action), newDepth, newAgentIndex)[0], action) 
                 for action in state.getLegalActions(agentIndex)]
        if agentIndex is 0:
            return max(scoreAction)
        else:
            return ( reduce(lambda acc,act: acc+act[0], scoreAction, 0)/float(len(scoreAction)) , )
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, self.depth)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsuleList =  currentGameState.getCapsules()
    score = currentGameState.getScore()
    newGhostStates = currentGameState.getGhostStates()
    scaredTimes = filter(lambda time: time > 0, [ghostState.scaredTimer for ghostState in newGhostStates])
    ghostPositions = currentGameState.getGhostPositions()
    for pos in ghostPositions:
        if manhattanDistance(pacmanPos, pos) < 2:
            score -= 1
    score += 1.0/min([manhattanDistance(pacmanPos, capsule) for capsule in capsuleList]) if len(capsuleList) > 0 else 0
    if len(scaredTimes) > 0:
        dist = manhattanDistance(pacmanPos, ghostPositions[0])
        if dist < 3:
            score += 200.0/dist
    score += 5.0/min([manhattanDistance(pacmanPos, food) for food in foodList]) if len(foodList) > 0 else  99999
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

