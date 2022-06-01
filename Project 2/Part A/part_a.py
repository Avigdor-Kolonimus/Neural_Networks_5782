import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import dirichlet

NEURONS_SET = 100
NEURONS_SMALL_SET = 30
DATA_SET = 300
LAST_ITERATION = 1
DEFAULT_ITERATIONS = 10
MAXSIZE = sys.maxsize
LOWER_BOUND = 0
UPPER_BOUND = 1
LOWER_RADIUS = 2
UPPER_RADIUS = 4

NEURON_COLOR = '#1c34d1'
POINT_COLOR = '#1cd1cb'
CIRCLE_COLOR = '#d11c58'
CHOSEN_POINT_COLOR = '#ff0000'
BMU_COLOR = '#e6ff00'

random.seed(47)

# -------------------------------------------------------------- Classes -------------------------------------------------
class Point:
    def __init__(self, x=0, y=0, chosen=0):
        """
        :param x: X Value
        :param y: Y Value
        :param chosen: Conscience
        """
        self.x = x
        self.y = y
        self.chosen = chosen


class Index:
    def __init__(self, x=0, y=0):
        """
        :param x: X Value
        :param y: Y Value
        """
        self.x = x
        self.y = y


class Node:
    def __init__(self, point=Point(), index=Index(), adjacent=[]):
        """
        :param point: Point (X, Y)
        :param index: The point index in the matrix topology.
        :param adjacent: The current point neighbours - matrix topology.
        """
        self.point = point
        self.index = index
        self.adjacent = adjacent

class KohonenAlgorithm:
    def __init__(self, set=NEURONS_SET, lowerBound=LOWER_BOUND, upperBound=UPPER_BOUND, learning_rate=0.5, scaling_constant=1e2, sigma=10, shape="Line", border=0, circleShape=0):
        if shape == "Line" or shape == "Circle":
            self.neurons = generateLine(set, lowerBound, upperBound)
        else:
            self.neurons = generateMatrix(lowerBound=lowerBound, upperBound=upperBound, set=set)

        self.shape = shape
        self.border = border
        self.circleShape = circleShape
        self.L0 = learning_rate  # the initial learning rate.
        self.lam = scaling_constant  # a time scaling constant.
        self.sigma0 = sigma  # the initial sigma.
    
    def train(self, data, iterations, title, uniform=0):
        """
        Function to activate the Kohonen algorithm 
        :param data: the data to be trained on.
        :param iterations: number of iterations.
        :param title: Title of the task.
        :param uniform: if 0 preforms uniform distribution sampling of the data. if 1/2 than non-uniform.
        :return:
        """
        iter_count = 0
        lenData = len(data)
        probability = self.getProbabilities(data, uniform)
        
        if self.shape == "Line" or self.shape == "Circle": 
            size = int(lenData/len(self.neurons)*2) + 1
            radius = int(len(self.neurons)/2)
            self.drow(data, title, chosenIndex=-1, bmuIndex=-1, border=self.border, circleShape=self.circleShape)
        else:
            size = int(lenData/len(self.neurons[0])*2) + 1
            radius = len(self.neurons[0]) + 1
            self.drow(data, title, chosenIndex=-1, bmuIndex=Index(-1, -1))

        while iter_count < iterations:
            if radius>2 and iter_count != 0 and iter_count % size == 0:
                radius -= 1

            chosenIndex = self.getRandomIndex(probability, lenData, uniform)
            bmuIndex = self.euclideanDist(data[chosenIndex])
            self.moveNeurons(data[chosenIndex], bmuIndex, iter_count, radius)
            
            iter_count += 1
            if iter_count % 500 == 0:
                self.drow(data, title + " Iter: " + str(iter_count), chosenIndex=chosenIndex, bmuIndex=bmuIndex, border=self.border, circleShape=self.circleShape)

        if self.shape == "Line" or self.shape == "Circle": 
            self.drow(data, title, chosenIndex=-1, bmuIndex=-1, border=self.border, circleShape=self.circleShape, done=LAST_ITERATION)
        else:
            self.drow(data, title, chosenIndex=-1, bmuIndex=Index(-1, -1), done=LAST_ITERATION)
    
    def getRandomIndex(self, probabilities, lenData, uniform):
        """
        Function returns probabilities for random choice
        :param probabilities: list of probabilities.
        :param lenData: range of data.
        :param uniform: if 0 preforms uniform distribution sampling of the data. if 1/2 than non-uniform.
        :return: random index
        """
        if uniform == 0:
            return np.random.choice(range(lenData))  # returns a random number.
        elif uniform == 1:
            return np.random.choice(range(lenData), p=probabilities)
        else:
            return np.random.choice(range(lenData), p=probabilities)
    
    def getProbabilities(self, data, uniform):
        """
        Function returns probabilities for random choice
        :param data: the data to be trained on.
        :param uniform: if 0 preforms uniform distribution sampling of the data. if 1/2 than non-uniform.
        :return: list of probabilities
        """
        if uniform == 1:
            return getDirichletProbabilities(len(data))
        elif uniform == 2:
            return getDistanceProbabilities(data)
        return []

    # the Euclidean distance between two points in Euclidean space is the length of a line segment between the two points.
    # https://en.wikipedia.org/wiki/Euclidean_distance
    def euclideanDist(self, data):
        """
        Function to find the minimum distance
        The Euclidean distance between two points in Euclidean space is the length of a line segment between the two points.
        https://en.wikipedia.org/wiki/Euclidean_distance
        :param data: The data to be trained on.
        :return: The closest neuron to the given point.
        """
        minimum = MAXSIZE
        if self.shape == "Line" or self.shape == "Circle": 
            index = 0
            lenNeurons = len(self.neurons)

            for i in range(lenNeurons):
                if self.neurons[i].chosen == 0:
                    distance = math.sqrt((data.x - self.neurons[i].x)**2 + (data.y - self.neurons[i].y)**2)
                    if distance < minimum:
                        minimum = distance
                        index = i
        else:
            index = Index()
            for i in range(len(self.neurons[0])):
                for j in range(len(self.neurons[0])):
                    if self.neurons[i][j].point.chosen == 0:
                        distance = math.sqrt((data.x - self.neurons[i][j].point.x) ** 2 + (data.y - self.neurons[i][j].point.y) ** 2)
                        if distance < minimum:
                            minimum = distance
                            index = Index(i, j)
        
        return index

    def updateConscience(self, index):
        """
        This function will reset the neurons conscience except for the closest neuron.
        :param index: Index of the closest neuron.
        :return: The updated neurons consciences.
        """
        if self.shape == "Line" or self.shape == "Circle": 
            for i in range(len(self.neurons)):
                if i != index:
                    self.neurons[i].chosen = 0
        else:
            for i in range(len(self.neurons[0])):
                for j in range(len(self.neurons[0])):
                    if i != index.x and j != index.y:
                        self.neurons[i][j].point.chosen = 0

    def moveNeurons(self, input_vector, index, t, radius):
        """
        Given neurons with topology of a line / circle, moving the adjacent neurons using Gaussian Distribution.
        :param input_vector: current data vector.
        :param index: Index of the closest neuron.
        :param t: Current time.
        :param radius: Number of adjacent neighbours.
        """
        if self.shape == "Line" or self.shape == "Circle": 
            self.moveNeuronAlgorithm(input_vector, index, t, radius)
        else:
            self.moveNeuronAlgorithmMatrix(input_vector, index, t, radius)

    def drow(self, data, title, chosenIndex, bmuIndex, border=0, circleShape=0, done=0):
        """
        Function to draw the points and neurons.
        :param points: Array of points.
        :param neurons: Array of neurons.
        :param title: Title of the task.
        :param chosenIndex: The chosen point.
        :param bmuIndex: The Best Matching Unit.
        :param border: Border = 0 -> draw rectangle border | Border = 1 -> draw ring border (2 circles).
        :param circleShape: circleShape = 0 -> line topology | circleShape = 1 -> circle topology.
        :param done: Done = 0 -> draw the board and clear | Done = 1 -> last iteration, show the board.
        """
        if self.shape == "Line" or self.shape == "Circle": 
            drowPaintNeurons(data, self.neurons, title, chosenIndex, bmuIndex, border, circleShape, done)
        else:
            drowPaintMatrix(data, self.neurons, title, chosenIndex, bmuIndex, done)
    # --------------------------------------------------------- Line / Circle ------------------------------------------------
    def moveNeuronAlgorithm(self, input_vector, index, t, radius):
        """
        Given neurons with topology of a line / circle, moving the adjacent neurons using Gaussian Distribution.
        :param input_vector: current data vector.
        :param index: Index of the closest neuron.
        :param t: Current time.
        :param radius: Number of adjacent neighbours.
        """
        lenNeurons = len(self.neurons)
        # The Best Matching Unit
        # The node with the smallest Euclidean difference between the input vector and all nodes is chosen, 
        # along with its neighbouring nodes within a certain radius, to have their position slightly adjusted to match the input vector.
        self.neurons[index].x += (input_vector.x - self.neurons[index].x) / 2
        self.neurons[index].y += (input_vector.y - self.neurons[index].y) / 2
        self.neurons[index].chosen = 1
 
        for i in range(1, radius + 1):
            i_right = index + i
            i_left = index - i
            # dist_to_bmu = np.linalg.norm((np.array((self.neurons[index].y, self.neurons[index].x)) - np.array((self.neurons[i].y, self.neurons[i].x))))
            delta = 1 / (2 ** (i + 1)) # the initial sigma.
        
            if i_right < lenNeurons:
                self.neurons[i_right].x += (input_vector.x - self.neurons[i_right].x) * delta * self.L(t)
                self.neurons[i_right].y += (input_vector.y - self.neurons[i_right].y) * delta * self.L(t)
                self.neurons[i_right].chosen = 1
        
            if i_left >= 0:
                self.neurons[i_left].x += (input_vector.x - self.neurons[i_left].x) * delta * self.L(t)
                self.neurons[i_left].y += (input_vector.y - self.neurons[i_left].y) * delta * self.L(t)
                self.neurons[i_left].chosen = 1 

        self.updateConscience(index)
    # -------------------------------------------------------------- Matrix --------------------------------------------------
    def moveNeuronAlgorithmMatrix(self, input_vector, index, t, radius):
        """
        Given neurons with topology of a 10x10, moving the adjacent neurons using Gaussian Distribution.
        :param input_vector: current data vector.
        :param index: Index of the closest neuron.
        :param t: Current time.
        :param radius: Number of adjacent neighbours.
        """
        self.neurons[index.x][index.y].point.x += (input_vector.x - self.neurons[index.x][index.y].point.x) / 2
        self.neurons[index.x][index.y].point.y += (input_vector.y - self.neurons[index.x][index.y].point.y) / 2
        self.neurons[index.x][index.y].point.chosen = 1

        counter = 0
        queue1 = []
        queue2 = []

        for i in range(len(self.neurons[index.x][index.y].adjacent)):
            queue1.append(self.neurons[index.x][index.y].adjacent[i])

        for i in range(1, radius + 1):
            counter += 1
            delta = 1 / (2 ** (i + 1)) # the initial sigma.


            if counter % 2 == 1:
                while len(queue1) > 0:
                    locate = queue1.pop(0)
                    node = self.neurons[locate.x][locate.y]
                    if node.point.chosen == 0:
                        self.neurons[locate.x][locate.y].point.x += (input_vector.x - node.point.x) * delta * self.L(t)
                        self.neurons[locate.x][locate.y].point.y += (input_vector.y - node.point.y) * delta * self.L(t)
                        self.neurons[locate.x][locate.y].point.chosen = 1
                        for k in range(len(node.adjacent)):
                            queue2.append(node.adjacent[k])

            if counter % 2 == 0:
                while len(queue2) > 0:
                    locate = queue2.pop(0)
                    node = self.neurons[locate.x][locate.y]
                    if node.point.chosen == 0:
                        self.neurons[locate.x][locate.y].point.x += (input_vector.x - node.point.x) * delta * self.L(t)
                        self.neurons[locate.x][locate.y].point.y += (input_vector.y - node.point.y) * delta * self.L(t)
                        self.neurons[locate.x][locate.y].point.chosen = 1
                        for k in range(len(node.adjacent)):
                            queue1.append(node.adjacent[k])

        self.updateConscience(index)


    def L(self, t):
        """
        Learning rate formula.
        t: current time.
        """
        return self.L0 * np.exp(-t / self.lam)

    def N(self, dist_to_bmu, t):
        """
        Computes the neighbouring penalty.
        dist_to_bmu: L2 distance to bmu.
        t: current time.
        """
        curr_sigma = self.sigma(t)
        return np.exp(-(dist_to_bmu ** 2) / (2 * curr_sigma ** 2))

    def sigma(self, t):
        """
        Neighbouring radius formula.
        t: current time.
        """
        return self.sigma0 * np.exp(-t / self.lam)

# ------------------------------------------------------------ Circle -----------------------------------------------------------
def generateCircle(radius1, radius2, set=DATA_SET):
    """
    Function generate the neuron circle.
    :param set: The number of neurons.
    :param radius1: Radius 1.
    :param radius2: Radius 2.
    :return: neurons
    """
    points = []

    for _ in range(set):
        x = random.uniform(-radius2, radius2)
        points.append(Point(x, generateCircleRing(x, radius1, radius2)))

    return points

def generateCircleRing(x, radius1, radius2=0):
    """
    Function to create points of data.
    Radius2 = 0 -> create points within a circle | Radius2 != 0 -> create points within a ring.
    :param x: Random X value
    :param radius1: Radius 1
    :param radius2: Radius 2
    :return: Random Y value within the circle / ring
    """
    if radius2 == 0:
        y_ = random.uniform(-radius1, radius1)
        while y_ ** 2 + x ** 2 > radius1 ** 2:
            y_ = random.uniform(-radius1, radius1)
        return y_
    else:
        y_ = random.uniform(-radius2, radius2)
        while (y_ ** 2 + x ** 2 > radius2 ** 2) or (y_ ** 2 + x ** 2 < radius1 ** 2):
            y_ = random.uniform(-radius2, radius2)
        return y_


# ------------------------------------------------------------ Matrix ------------------------------------------------------------
def createTwoDimensionalArray(neurons, isqrt):
    """
    :param neurons: isqrtXisqrt neurons.
    :param isqrt: The number of neurons in one row/column.
    :return: Neurons arranged in a isqrtXisqrt topology.
    """
    matrix = [[Node() for i in range(isqrt)] for j in range(isqrt)]

    "Corners"
    matrix[0][0] = Node(neurons[0][0], Index(0, 0), [Index(0, 1), Index(1, 0)])
    matrix[0][isqrt-1] = Node(neurons[0][isqrt-1], Index(0, 4), [Index(0, 3), Index(1, 4)])
    matrix[isqrt-1][0] = Node(neurons[isqrt-1][0], Index(4, 0), [Index(3, 0), Index(4, 1)])
    matrix[isqrt-1][isqrt-1] = Node(neurons[isqrt-1][isqrt-1], Index(4, 4), [Index(3, 4), Index(4, 3)])

    for i in range(1, isqrt-1):
        "Edges"
        matrix[0][i] = Node(neurons[0][i], Index(0, i), [Index(0, i - 1), Index(1, i), Index(0, i + 1)])
        matrix[i][0] = Node(neurons[i][0], Index(i, 0), [Index(i - 1, 0), Index(i, 1), Index(i + 1, 0)])
        matrix[isqrt-1][i] = Node(neurons[isqrt-1][i], Index(4, i), [Index(4, i - 1), Index(3, i), Index(4, i + 1)])
        matrix[i][isqrt-1] = Node(neurons[i][isqrt-1], Index(i, 4), [Index(i - 1, 4), Index(i, 3), Index(i + 1, 4)])

        "General Case"
        for j in range(1, isqrt-1):
            matrix[i][j] = Node(neurons[i][j], Index(i, j), [Index(i, j-1), Index(i-1, j), Index(i, j+1), Index(i+1, j)])

    return matrix

def generateMatrix(lowerBound , upperBound, set=NEURONS_SET):
    """
    Function generate the neuron matrix.
    :param set: The number of neurons.
    :param lowerBound: The lower bound for a lower parameter in random.uniform function.
    :param upperBound: The upper bound for a high parameter in random.uniform function.
    :return: neurons
    """
    isqrt = math.isqrt(set)
    neurons = []
    neurons = [[Point() for i in range(isqrt)] for j in range(isqrt)]
    for i in range(isqrt):
        for j in range(isqrt):
            neurons[i][j] = Point(random.uniform(lowerBound, upperBound), random.uniform(lowerBound, upperBound))
    matrix = createTwoDimensionalArray(neurons, isqrt)

    return matrix

# --------------------------------------------------------- Drow Function --------------------------------------------------------
def drowPaintNeurons(points, neurons, title, chosenIndex=-1, bmuIndex=-1, border=0, circleShape=0, done=0):
    """
    Function to draw the points and neurons.
    :param points: Array of points.
    :param neurons: Array of neurons.
    :param title: Title of the task.
    :param chosenIndex: The chosen point.
    :param bmuIndex: The Best Matching Unit.
    :param border: Border = 0 -> draw rectangle border | Border = 1 -> draw ring border (2 circles).
    :param circleShape: circleShape = 0 -> line topology | circleShape = 1 -> circle topology.
    :param done: Done = 0 -> draw the board and clear | Done = 1 -> last iteration, show the board.
    :return: None
    """
    neurons_x = []
    neurons_y = []

    for i in range(len(points)):
        if done == 0 and chosenIndex == i:
            plt.scatter(points[i].x, points[i].y, color=CHOSEN_POINT_COLOR, label='Chosen data point')
        else:
            plt.scatter(points[i].x, points[i].y, color=POINT_COLOR)
    
    for i in range(len(neurons)):
        neurons_x.append(neurons[i].x)
        neurons_y.append(neurons[i].y)
        if done == 0 and bmuIndex == i:
            plt.scatter(neurons[i].x, neurons[i].y, color=BMU_COLOR, label='BMU')
        else:
            plt.scatter(neurons[i].x, neurons[i].y, color=NEURON_COLOR)
    if circleShape == 1:
        neurons_x.append(neurons_x[0]), neurons_y.append(neurons_y[0])

    if border == 0:
        # {(x,y) |  0 <= x <= 1, 0<=y<=1}
        rectangle = plt.Rectangle((0,0), 1, 1,  color=CIRCLE_COLOR, fill=False)
        ax = plt.gca()
        ax.add_patch(rectangle)
    elif border == 1:
        circle1 = plt.Circle((0, 0), math.sqrt(LOWER_RADIUS), color=CIRCLE_COLOR, fill=False)
        circle2 = plt.Circle((0, 0), math.sqrt(UPPER_RADIUS), color=CIRCLE_COLOR, fill=False)
        ax = plt.gca()
        ax.add_patch(circle1)
        ax.add_patch(circle2)

    plt.suptitle(title)
    plt.plot(neurons_x, neurons_y)

    if done == LAST_ITERATION:
        plt.show()
    else:
        plt.legend(loc="upper left")
        plt.draw()
        plt.pause(0.01)
        plt.clf()


def drowPaintMatrix(points, matrix, title, chosenIndex=-1, bmuIndex=Index(-1, -1), done=0):
    """
    Function to draw the points and matrix.
    :param points: Array of points.
    :param matrix: Matrix of neurons.
    :param title: Title of the task.
    :param chosenIndex: The chosen point.
    :param bmuIndex: The Best Matching Unit.
    :param done: Done = 0 -> draw the matrix and clear | Done = 1 -> last iteration, show the matrix.
    :return: None
    """
    neurons_x = [[] for i in range(2 * len(matrix[0]))]
    neurons_y = [[] for i in range(2 * len(matrix[0]))]
    
    for i in range(len(points)):
        if done == 0 and chosenIndex == i:
            plt.scatter(points[i].x, points[i].y, color=CHOSEN_POINT_COLOR, label='Chosen data point')
        else:
            plt.scatter(points[i].x, points[i].y, color=POINT_COLOR)
    
    index = 0
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if done == 0 and bmuIndex.x == i and bmuIndex.y == j:
                plt.scatter(matrix[i][j].point.x, matrix[i][j].point.y, color=BMU_COLOR, label='BMU')
            else:
                plt.scatter(matrix[i][j].point.x, matrix[i][j].point.y, color=NEURON_COLOR)
            neurons_x[index].append(matrix[i][j].point.x)
            neurons_y[index].append(matrix[i][j].point.y)
        index += 1
    
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            neurons_x[index].append(matrix[j][i].point.x)
            neurons_y[index].append(matrix[j][i].point.y)
        index += 1

    plt.suptitle(title)
    
    for i in range(len(neurons_x)):
        plt.plot(neurons_x[i], neurons_y[i], NEURON_COLOR)
    
    # {(x,y) |  0 <= x <= 1, 0<=y<=1}
    rectangle = plt.Rectangle((0,0), 1, 1,  color=CIRCLE_COLOR, fill=False)
    ax = plt.gca()
    ax.add_patch(rectangle)
    
    if done == LAST_ITERATION:
        plt.show()
    else:
        plt.legend(loc="upper left")
        plt.draw()
        plt.pause(0.01)
        plt.clf()

# --------------------------------------------------------- Help Function --------------------------------------------------------
# https://en.wikipedia.org/wiki/Dirichlet_distribution
def getDirichletProbabilities(length):
    """
    Function generate the probability array by the dirichlet function.
    :param length: lenght of array.
    :return: probability array
    """
    probability = dirichlet([1] * length)  # uses the dirichlet function to distribute probabilities.
    return probability

def getDistanceProbabilities(points):
    """
    Function generate the probability of a point being chosen as a data point is proportional to the distance from the center of the disk.
    :param points: Array of points.
    :return: probability array
    """
    probability = []
    xCenter = (UPPER_BOUND - LOWER_BOUND)/2
    yCenter = xCenter

    for point in points:
        probability.append(math.dist([point.x, point.y], [xCenter, yCenter]))
    
    probability = np.asarray(probability)
    probability = (probability - min(probability)) / sum(probability - min(probability))
    return probability

def generateLine(set, lowerBound , upperBound):
    """
    Function generate the neuron line.
    :param set: The number of neurons.
    :param lowerBound: The lower bound for a lower parameter in random.uniform function.
    :param upperBound: The upper bound for a high parameter in random.uniform function.
    :return: neurons
    """
    neurons = []
    y = (upperBound - lowerBound)/2
    for _ in range(set):
        neurons.append(Point(random.uniform(lowerBound, upperBound), y))
    return neurons

def generateSquare(set, lowerBound , upperBound):
    """
    Function generate the data points.
    :param set: The number of points.
    :param lowerBound: The lower bound for a lower parameter in random.uniform function.
    :param upperBound: The upper bound for a high parameter in random.uniform function.
    :return: The data points
    """
    squareData = []

    # the data set is {(x,y) |  0 <= x <= 1, 0<=y<=1}
    for i in range(set):
        squareData.append(Point(random.uniform(lowerBound, upperBound), random.uniform(lowerBound, upperBound)))
    return squareData

def main():
    lowerBound = LOWER_BOUND
    upperBound = UPPER_BOUND
    # Part A.1
    # -------------------------------------------------------- Uniform Dat ---------------------------------------------------------
    # ------------------------------------------------------- Line Topology --------------------------------------------------------
    # create points
    points = generateSquare(set=500, lowerBound=lowerBound, upperBound=upperBound)
    
    # 2500 iterations
    somSquare = KohonenAlgorithm(set=NEURONS_SET, lowerBound=lowerBound, upperBound=upperBound, shape="Line", border=0, circleShape=0)
    iterations = 2500
    somSquare.train(points, iterations, "Line Topology - Uniform Data", uniform=0)
    
    # ------------------------------------------------------- 10x10 Topology -------------------------------------------------------
    # 2500 iterations
    somSquare = KohonenAlgorithm(set=NEURONS_SET, lowerBound=lowerBound, upperBound=upperBound, shape="Matrix")
    iterations = 2500
    somSquare.train(points, iterations, "10x10 Topology - Uniform Data", uniform=0)

    # Part A.2
    # ------------------------------------------------------ Non-uniform Dat -------------------------------------------------------
    # ------------------------------------------------------- Line Topology --------------------------------------------------------
    points = generateSquare(set=500, lowerBound=lowerBound, upperBound=upperBound)
    
    # 2500 iterations
    somSquare = KohonenAlgorithm(set=NEURONS_SET, lowerBound=lowerBound, upperBound=upperBound, shape="Line", border=0, circleShape=0)
    iterations = 2500
    somSquare.train(points, iterations, "Line Topology - Non-uniform Data (type 1)", uniform=1)

    # 2500 iterations
    somSquare = KohonenAlgorithm(set=NEURONS_SET, lowerBound=lowerBound, upperBound=upperBound, shape="Line", border=0, circleShape=0)
    iterations = 2500
    somSquare.train(points, iterations, "Line Topology - Non-uniform Data (type 2)", uniform=2)
    
    # ------------------------------------------------------- 10x10 Topology -------------------------------------------------------
    # 2500 iterations
    somSquare = KohonenAlgorithm(set=NEURONS_SET, lowerBound=lowerBound, upperBound=upperBound, shape="Matrix")
    iterations = 2500
    somSquare.train(points, iterations, "10x10 Topology - Non-uniform (type 1)", uniform=1)

    # 2500 iterations
    somSquare = KohonenAlgorithm(set=NEURONS_SET, lowerBound=lowerBound, upperBound=upperBound, shape="Matrix")
    iterations = 2500
    somSquare.train(points, iterations, "10x10 Topology - Non-uniform (type 2)", uniform=2)
    # Part A.3
    # ------------------------------------------------------- Circle Topology ------------------------------------------------------
    points = generateCircle(math.sqrt(LOWER_RADIUS), math.sqrt(UPPER_RADIUS), set=DATA_SET)
    
    # 2500 iterations
    somSquare = KohonenAlgorithm(set=NEURONS_SMALL_SET, lowerBound=lowerBound, upperBound=upperBound, shape="Circle", border=1, circleShape=1)
    iterations = 2500
    somSquare.train(points, iterations, "Circle Topology - Uniform Data", uniform=0)


if __name__ == '__main__':
    main()