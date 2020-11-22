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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

        remainingFood = newFood.asList()
        currentFood = currentGameState.getFood().asList()

        score = successorGameState.getScore()
        foodPos = []
        ghostPos = []

        #calculamos la distancia de manhattan de la posición del pacman a la comida que queda
        for food in remainingFood:
            foodPos.append(manhattanDistance(newPos, food))

        #hacemos lo mismo con los fantasmas
        for ghost in newGhostStates:
            ghostPos.append(manhattanDistance(newPos, ghost.configuration.getPosition()))

        #si queda comida, calculamos cuál está más cerca y vamos hacia él
        if foodPos:
            closest = min(foodPos)
            if newPos not in currentFood:
                score -= closest

        if ghostPos:
            closest = min(ghostPos)
            scared = True
            for ghostStatus in newScaredTimes:
                #eso significa que el fantasma no está asustado
                #por tanto debemos alejarnos mucho de él
                if ghostStatus == 0:
                    scared = False
                    break
            #como queremos alejarnos, si está muy cerca y no asustado, bajamos la puntuación
            if scared == False:
                if closest < 2:
                    score = -9999
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def maxFunction(gameState, depth, agentIndex):
            #cada vez que llamemos a esta función iremos un
            depth -= 1
            #comprovamos si hemos llegado a un estado terminal
            if depth == 0 or gameState.isLose() or gameState.isWin():
                #en ese caso, calculamos la función de evaluación y lo devolvemos
                return (self.evaluationFunction(gameState),None)

            #inicializamos la puntuación maxima a -inf para que nos haga bien el máximo
            maxScore = float("-inf")

            #por cada acción que podamos hacer, cogemos los sucesores y, siguendo el algoritmo
            #invocamos la función minFunction para encontrar la de coste mínimo
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                minScore = minFunction(successor, depth, agentIndex + 1)[0]
                #con este if nos aseguramos que cogemos el màximo
                if minScore > maxScore:
                    maxScore = minScore
                    maxAction = action
            return (maxScore, maxAction)

        #para la función de mínimo hacemos un procedimiento similar
        def minFunction(gameState, depth, agentIndex):
            #comprovamos si el estado es terminal
            if gameState.isLose() or gameState.isWin():
                #en ese caso calculamos la función de evaluación del estado
                return (self.evaluationFunction(gameState),None)

            #inicializamos la puntuación minima a inf para que nos haga bien el minimo
            minScore = float("inf")

            #ahora debemos estudiar qué nos conviene más, si maxFunction o minFunction
            #en el caso de que tengamos un agentIndex menor q el numero de agentes-1,
            #es mejor utilizar el minFunction otra vez
            if(agentIndex < gameState.getNumAgents() - 1):
                metodName, newAgentIndex = (minFunction, agentIndex + 1)
            #en caso contrario, hacemos maxFunction normal
            else:
                metodName, newAgentIndex = (maxFunction,0)

            #el for es análogo al de maxFunction
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                score = metodName(successor, depth, newAgentIndex)[0]
                if score < minScore:
                    minScore = score
                    minAction = action
            return (minScore, minAction)

        #finalmente, devolvemos la acción final máxima
        return maxFunction(gameState, self.depth, 0)[1]





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def maxFunction(gameState, depth, agentIndex, alpha, beta):
            #cada vez que llamemos a esta función iremos un
            depth -= 1
            #comprovamos si hemos llegado a un estado terminal
            if depth < 0 or gameState.isLose() or gameState.isWin():
                #en ese caso, calculamos la función de evaluación y lo devolvemos
                return (self.evaluationFunction(gameState),None)

            #inicializamos la puntuación maxima a -inf para que nos haga bien el máximo
            maxScore = float("-inf")

            #por cada acción que podamos hacer, cogemos los sucesores y, siguendo el algoritmo
            #invocamos la función minFunction para encontrar la de coste mínimo
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                minScore = minFunction(successor, depth, agentIndex + 1, alpha, beta)[0]
                #con este if nos aseguramos que cogemos el màximo
                if minScore > maxScore:
                    maxScore = minScore
                    maxAction = action

                #comprovamos si nuestra score es mayor que beta
                if maxScore > beta:
                    #en ese caso devolvemos directamente la score y la acción
                    return (maxScore, maxAction)

                #tal como indica el algoritmo, alpha = maximo entre la acción y alpha
                alpha = max(alpha, maxScore)
            return (maxScore, maxAction)

        #para la función de mínimo hacemos un procedimiento similar
        def minFunction(gameState, depth, agentIndex, alpha, beta):
            #comprovamos si el estado es terminal
            if gameState.isLose() or gameState.isWin():
                #en ese caso calculamos la función de evaluación del estado
                return (self.evaluationFunction(gameState),None)

            #inicializamos la puntuación minima a inf para que nos haga bien el minimo
            minScore = float("inf")

            #ahora debemos estudiar qué nos conviene más, si maxFunction o minFunction
            #en el caso de que tengamos un agentIndex menor q el numero de agentes-1,
            #es mejor utilizar el minFunction otra vez
            if(agentIndex < gameState.getNumAgents() - 1):
                methodName, newAgentIndex = (minFunction, agentIndex + 1)
            #en caso contrario, hacemos maxFunction normal
            else:
                methodName, newAgentIndex = (maxFunction,0)

            #el for es análogo al de maxFunction
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                score = methodName(successor, depth, newAgentIndex, alpha, beta)[0]
                if score < minScore:
                    minScore = score
                    minAction = action
                #comprovamos que el score de la acción es menor que alpha
                if minScore < alpha:
                    #en ese caso lo devolvemos
                    return (minScore, minAction)
                #tal como indica el algoritmo, cogemos beta = minimo entre beta y la acción
                beta = min(beta, minScore)
            return (minScore, minAction)

        #finalmente, devolvemos la acción final máxima
        return maxFunction(gameState, self.depth, 0, float("-inf"), float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"


        def value(gameState, depth, agentIndex):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState), Directions.STOP)
            else:
                if agentIndex == gameState.getNumAgents():
                    agentIndex = 0
                if agentIndex == gameState.getNumAgents() - 1:
                    depth = depth - 1
                if agentIndex == 0:
                    return maxValue(gameState, depth, agentIndex)
                else:
                    return expValue(gameState, depth, agentIndex)

        def maxValue(gameState, depth, agentIndex):
            v = -99999999999
            vaction = gameState.getLegalActions(agentIndex)[0]
            for action in gameState.getLegalActions(agentIndex):
                v_point, v_action = value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
                if v_point > v:
                    v = v_point
                    vaction = action
            return (v, vaction)

        def expValue(gameState, depth, agentIndex):
            score = 0
            vaction = gameState.getLegalActions(agentIndex)[0]
            for action in gameState.getLegalActions(agentIndex):
                v_point, v_action = value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
                score += v_point * 1 / len(gameState.getLegalActions(agentIndex))
            return (score, vaction)

        return value(gameState, self.depth, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
