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
                    score = float("-inf")
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
            if depth < 0 or gameState.isLose() or gameState.isWin():
                #en ese caso, calculamos la función de evaluación y lo devolvemos
                return (self.evaluationFunction(gameState),None)

            #inicializamos la puntuación maxima a -inf para que nos haga bien el máximo
            maxValue = float("-inf")

            #por cada acción que podamos hacer, cogemos los sucesores y, siguendo el algoritmo
            #invocamos la función minFunction para encontrar la de coste mínimo
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = minFunction(successor, depth, agentIndex + 1)[0]
                #con este if nos aseguramos que cogemos el màximo
                if value > maxValue:
                    maxValue = value
                    maxAction = action
            return (maxValue, maxAction)

        #para la función de mínimo hacemos un procedimiento similar
        def minFunction(gameState, depth, agentIndex):
            #comprovamos si el estado es terminal
            if gameState.isLose() or gameState.isWin():
                #en ese caso calculamos la función de evaluación del estado
                return (self.evaluationFunction(gameState),None)

            #inicializamos la puntuación minima a inf para que nos haga bien el minimo
            minValue = float("inf")

            #agentIndex < numero de agentes -1 significa que todavía no estamos en el último nivel
            #del árbol por tanto, podemos seguir minimizando las acciones de los agents
            if(agentIndex < gameState.getNumAgents() - 1):
                chosenMethod, newAgentIndex = (minFunction, agentIndex + 1)
            #en caso contrario, pasamos a maximizar el movimiento del pacman
            else:
                chosenMethod, newAgentIndex = (maxFunction, 0)

            #el for es análogo al de maxFunction
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = chosenMethod(successor, depth, newAgentIndex)[0]
                if value < minValue:
                    minValue = value
                    minAction = action
            return (minValue, minAction)

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
            maxValue = float("-inf")

            #por cada acción que podamos hacer, cogemos los sucesores y, siguendo el algoritmo
            #invocamos la función minFunction para encontrar la de coste mínimo
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = minFunction(successor, depth, agentIndex + 1, alpha, beta)[0]
                #con este if nos aseguramos que cogemos el màximo
                if value > maxValue:
                    maxValue = value
                    maxAction = action

                #comprovamos si nuestra score es mayor que beta
                if maxValue > beta:
                    #en ese caso devolvemos directamente la score y la acción
                    return (maxValue, maxAction)

                #tal como indica el algoritmo, alpha = maximo entre la acción y alpha
                alpha = max(alpha, maxValue)
            return (maxValue, maxAction)

        #para la función de mínimo hacemos un procedimiento similar
        def minFunction(gameState, depth, agentIndex, alpha, beta):
            #comprovamos si el estado es terminal
            if gameState.isLose() or gameState.isWin():
                #en ese caso calculamos la función de evaluación del estado
                return (self.evaluationFunction(gameState),None)

            #inicializamos la puntuación minima a inf para que nos haga bien el minimo
            minValue = float("inf")

            #agentIndex < numero de agentes -1 significa que todavía no estamos en el último nivel
            #del árbol por tanto, podemos seguir minimizando las acciones de los agentes
            if(agentIndex < gameState.getNumAgents() - 1):
                methodName, newAgentIndex = (minFunction, agentIndex + 1)
            #en caso contrario, pasamos a maximizar el movimiento del pacman
            else:
                methodName, newAgentIndex = (maxFunction,0)

            #el for es análogo al de maxFunction
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = methodName(successor, depth, newAgentIndex, alpha, beta)[0]
                if value < minValue:
                    minValue = value
                    minAction = action
                #comprovamos que el score de la acción es menor que alpha
                if minValue < alpha:
                    #en ese caso lo devolvemos
                    return (minValue, minAction)
                #tal como indica el algoritmo, cogemos beta = minimo entre beta y la acción
                beta = min(beta, minValue)
            return (minValue, minAction)

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
        return self.value(gameState, 0, 0)

    #vamos a hacer las tres funciones separadas, value, maxFunction y expectFunction
    def value(self, gameState, depth, agentIndex):
        #primero siempre comprovamos si es un estado final o no
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        #si el agentIndex = 0, significa que el pacman es óptimo
        #por tanto, debemos aplicar al gameState la función maxValue
        if (agentIndex == 0):
            return self.maxFunction(gameState, depth, agentIndex)

        #por lo contrario, en este caso el fantasma es óptimo y aplicaremos expectValue
        else:
            return self.expectFunction(gameState, depth, agentIndex)


    def maxFunction(self, gameState, depth, agentIndex):
        #incializamos el valor a -inf para que haga correctamente el máximo
        maxValue = float('-inf')

        #comprovamos si es un estado final y en ese caso calculamos la score del gameState
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        #por cada acción, miramos sus hijos y nos quedamos con el que nos de valor máximo
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.expectFunction(successor, depth, agentIndex + 1)

            #hacemos el if correspondiente a la búsqueda del valor máximo
            if value > maxValue:
                maxValue = value
                maxAction = action

        #debemos tener en cuenta cuando el estado llega a él mismo (en ese caso la depth = 0)
        if depth == 0:
            return maxAction
        #si no es el caso, enviamos el maxValue como siempre
        else:
            return maxValue

    def expectFunction(self, gameState, depth, agentIndex):
        #inicalizamos el value a 0 para poder hacer sumas luego
        value = 0
        #comprovamos que no sea estado terimal
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agentIndex):
            #si el agentIndex corresponde a la última posición de la lista de agentes, debemos ir al siguiente
            #nivel del árbol y continuar buscando el valor máximo del pacman
            if agentIndex == gameState.getNumAgents() - 1:
                successor = gameState.generateSuccessor(agentIndex, action)
                value += self.maxFunction(successor, depth + 1, 0)

            #en caso contrario, continuamos con la función expect
            else:
                successor = gameState.generateSuccessor(agentIndex, action)
                value += self.expectFunction(successor, depth, agentIndex + 1)

        #devovemos la probabilidad del resultado
        return value / len(gameState.getLegalActions(agentIndex))



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: la intención de esta función es encontrar el agente más cercano,
    la capsule más cercana y la comida restante y combinar estos tres elementos
    para conseguir una buena función de evaluación para que el pacman se mueva correctamente
    """
    currentGameScore = scoreEvaluationFunction(currentGameState)

    #agente más cercano:
    #inicializamos las variables que vamos a necesitar
    currentGhostStates = currentGameState.getGhostStates()
    pacmanPosition = currentGameState.getPacmanPosition()
    minGhostDistance = float("inf")

    #iremos recorriendo la lista de todos para ver cuál es el más cercano
    for ghost in currentGhostStates:
        #lo que determinará la distancia será la distáncia de manhattan
        ghostDistance = manhattanDistance(pacmanPosition, ghost.getPosition())
        if ghostDistance < minGhostDistance:
            minGhostDistance = ghostDistance

    #hacemos el if para evitar dividir entre zero
    if minGhostDistance == 0:
        minGhostDistance = 1


    #capsule más cercana:
    #inicializamos las variables que vamos a necesitar
    pacmanPosition = currentGameState.getPacmanPosition()
    currentCapsules = currentGameState.getCapsules()
    minCapsuleDistance = 0

    #comprovamos si hay alguna
    if (currentCapsules):
        #en ese caso, lo inicializamos para que haga bien el mínimo
        minCapsuleDistance = float("inf")

        #iremos recorriendolas todas para hacer el mínimo
        for capsule in currentCapsules:
            capsuleDistance = manhattanDistance(pacmanPosition,capsule)
            if capsuleDistance < minCapsuleDistance:
                minCapsuleDistance = capsuleDistance


    #finalmente calculamos el average de la comida que queda por comer
    #inicializamos las variables que vamos a usar
    currentFood = currentGameState.getFood()
    currentPosition = currentGameState.getPacmanPosition()
    foodDistance = []

    for x, row in enumerate(currentFood):
        for y, column in enumerate(currentFood[x]):
            if currentFood[x][y]:
                foodDistance.append(manhattanDistance(currentPosition, (x,y)))
    avg = sum(foodDistance)/float(len(foodDistance)) if (foodDistance and sum(foodDistance) != 0) else 1


    return currentGameScore - avg + 2.0/minGhostDistance - currentGameState.getNumFood() - minCapsuleDistance

# Abbreviation
better = betterEvaluationFunction
