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


# --------------------------------------------------------- Line / Circle ------------------------------------------------
def updateConscience(index, neurons):
    """
    This function will reset the neurons conscience except for the closest neuron.
    :param index: Index of the closest neuron
    :param neurons: Array of neurons
    :return: The updated neurons consciences.
    """
    for i in range(len(neurons)):
        if i != index:
            neurons[i].chosen = 0
    return neurons

# the Euclidean distance between two points in Euclidean space is the length of a line segment between the two points.
# https://en.wikipedia.org/wiki/Euclidean_distance
def euclideanDist(point, neurons):
    """
    Function to find the minimum distance
    The Euclidean distance between two points in Euclidean space is the length of a line segment between the two points.
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param point: Given Point
    :param neurons: Array of neurons
    :return: The closest neuron to the given point.
    """
    minimum = MAXSIZE
    index = 0

    for i in range(len(neurons)):
        if neurons[i].chosen == 0:
            distance = math.sqrt((point.x - neurons[i].x)**2 + (point.y - neurons[i].y)**2)
            if distance < minimum:
                minimum = distance
                index = i
    return index


def moveNeuronAlgorithm(point, index, neurons, radius):
    """
    Given neurons with topology of a line / circle, moving the adjacent neurons using Gaussian Distribution.
    :param point: Given Point
    :param index: Index of the closest neuron
    :param neurons: Array of neurons
    :param radius: Number of adjacent neighbours.
    :return: The updated neurons locations according to the given radius.
    """
    # The Best Matching Unit
    # The node with the smallest Euclidean difference between the input vector and all nodes is chosen, 
    # along with its neighbouring nodes within a certain radius, to have their position slightly adjusted to match the input vector.
    neurons[index].x += (point.x - neurons[index].x) / 2
    neurons[index].y += (point.y - neurons[index].y) / 2
    neurons[index].chosen = 1
 
    for i in range(1, radius + 1):
        i_right = index + i
        i_left = index - i
        delta = 1 / (2 ** (i + 1)) # the initial sigma.
        
        if i_right < len(neurons):
            neurons[i_right].x += (point.x - neurons[i_right].x) * delta
            neurons[i_right].y += (point.y - neurons[i_right].y) * delta
            neurons[i_right].chosen = 1
        
        if i_left >= 0:
            neurons[i_left].x += (point.x - neurons[i_left].x) * delta
            neurons[i_left].y += (point.y - neurons[i_left].y) * delta
            neurons[i_left].chosen = 1

    return updateConscience(index, neurons)


def drowPaintNeurons(points, neurons, title, chosenIndex=-1, bmuIndex=-1, border=0, circleShape=0, done=0):
    """
    Function to draw the points and neurons.
    :param points: Array of points.
    :param neurons: Array of neurons.
    :param title: Title of the task.
    :param chosenIndex: The chosen point.
    :param bmuIndex: The Best Matching Unit.
    :param border: Border = 0 -> draw circle border | Border = 1 -> draw ring border (2 circles).
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
        xCircle = (LOWER_BOUND + UPPER_BOUND)/2
        yCircle = xCircle
        radiusCircle = math.sqrt(LOWER_BOUND + UPPER_BOUND)
        circle1 = plt.Circle((xCircle, yCircle), radiusCircle, color=CIRCLE_COLOR, fill=False)
        ax = plt.gca()
        ax.add_patch(circle1)
        # {(x,y) |  0 <= x <= 1, 0<=y<=1}, rectangle ?
        # rectangle = plt.Rectangle((0,0), 1, 1,  color=CIRCLE_COLOR, fill=False)
        # ax = plt.gca()
        # ax.add_patch(rectangle)
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


def algorithm(points, neurons, uniform=0, iterations=DEFAULT_ITERATIONS, title="", border=0, circleShape=0):
    """
    Function to activate the Kohonen algorithm 
    :param points: Array of points.
    :param neurons: Array of neurons.
    :param iterations: Number of train iterations.
    :param uniform: if 0 preforms uniform distribution sampling of the data.
    :param title: Title of the task.
    :param border: Border = 0 -> draw circle border | Border = 1 -> draw ring border (2 circles)
    :param circleShape: circleShape = 0 -> line topology | circleShape = 1 -> circle topology.
    :return: The updated neurons locations according to the given radius.
    """
    lenPoints = len(points)
    size = int(lenPoints/len(neurons)*2) + 1
    probability= []

    if uniform == 1:
        probability = getDirichletProbabilities(lenPoints)
    elif uniform == 2:
        probability = getDistanceProbabilities(points)

    for j in range(iterations):
        if uniform == 0:
            random.shuffle(points)
        radius = int(len(neurons)/2)
        for i in range(lenPoints):
            if i % size == 0 and i != 0:
                radius -= 1
            
            if uniform == 0:
                chosenIndex = np.random.choice(range(lenPoints))  # returns a random number.
            elif uniform == 1:
                chosenIndex = np.random.choice(range(lenPoints), p=probability)
            elif uniform == 2:
                chosenIndex = np.random.choice(range(lenPoints), p=probability)

            # chosenIndex = random.randint(0, lenPoints-1)
            bmuIndex = euclideanDist(points[chosenIndex], neurons)
            neurons = moveNeuronAlgorithm(points[chosenIndex], bmuIndex, neurons, radius)

            iter = j * lenPoints + i
            if iter % lenPoints == 0:
                drowPaintNeurons(points, neurons, title + " Iter: " + str(iter), chosenIndex=chosenIndex, bmuIndex=bmuIndex, border=border, circleShape=circleShape)

    return neurons

# -------------------------------------------------------------- Matrix -------------------------------------------------
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


def euclideanDistMatrix(point, matrix):
    """
    Function to find the minimum distance
    The Euclidean distance between two points in Euclidean space is the length of a line segment between the two points.
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param point: Given Point
    :param matrix: Matrix of neurons
    :return: The closest neuron to the given point.
    """
    minimum = MAXSIZE
    index = Index()
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if matrix[i][j].point.chosen == 0:
                distance = math.sqrt((point.x - matrix[i][j].point.x) ** 2 + (point.y - matrix[i][j].point.y) ** 2)
                if distance < minimum:
                    minimum = distance
                    index = Index(i, j)
    return index


def updateConscienceMatrix(index, matrix):
    """
    This function will reset the neurons conscience except for the closest neuron for 10x10 topology.
    :param index: Index of the closest neuron
    :param matrix: Matrix of neurons
    :return: The updated neurons consciences.
    """
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if i != index.x and j != index.y:
                matrix[i][j].point.chosen = 0
    return matrix


def moveNeuroAlgorithmMatrix(point, index, matrix, radius):
    """
    Given neurons with topology of a 10x10, moving the adjacent neurons using Gaussian Distribution.
    :param point: Given Point
    :param index: Index of the closest neuron
    :param matrix: Matrix of neurons
    :param radius: Number of adjacent neighbours.
    :return: The updated matrix locations according to the given radius.
    """
    matrix[index.x][index.y].point.x += (point.x - matrix[index.x][index.y].point.x) / 2
    matrix[index.x][index.y].point.y += (point.y - matrix[index.x][index.y].point.y) / 2
    matrix[index.x][index.y].point.chosen = 1

    counter = 0
    queue1 = []
    queue2 = []

    for i in range(len(matrix[index.x][index.y].adjacent)):
        queue1.append(matrix[index.x][index.y].adjacent[i])

    for i in range(1, radius + 1):
        counter += 1
        delta = 1 / (2 ** (i + 1))
        if counter % 2 == 1:
            while len(queue1) > 0:
                locate = queue1.pop(0)
                node = matrix[locate.x][locate.y]
                if node.point.chosen == 0:
                    matrix[locate.x][locate.y].point.x += (point.x - node.point.x) * delta
                    matrix[locate.x][locate.y].point.y += (point.y - node.point.y) * delta
                    matrix[locate.x][locate.y].point.chosen = 1
                    for k in range(len(node.adjacent)):
                        queue2.append(node.adjacent[k])

        if counter % 2 == 0:
            while len(queue2) > 0:
                locate = queue2.pop(0)
                node = matrix[locate.x][locate.y]
                if node.point.chosen == 0:
                    matrix[locate.x][locate.y].point.x += (point.x - node.point.x) * delta
                    matrix[locate.x][locate.y].point.y += (point.y - node.point.y) * delta
                    matrix[locate.x][locate.y].point.chosen = 1
                    for k in range(len(node.adjacent)):
                        queue1.append(node.adjacent[k])

    return updateConscienceMatrix(index, matrix)


def algorithmMatrix(points, matrix, uniform=0, iterations=DEFAULT_ITERATIONS, title=""):
    """
    Function to activate the Kohonen algorithm for matrix
    :param points: Array of points.
    :param matrix: Matrix of neurons.
    :param uniform: if 0 preforms uniform distribution sampling of the data.
    :param iterations: Number of train iterations.
    :param title: Title of the task.
    :return: None
    """
    lenPoints = len(points)
    radius = len(matrix[0]) + 1
    probability= []

    if uniform == 1:
        probability = getDirichletProbabilities(lenPoints)
    elif uniform == 2:
        probability = getDistanceProbabilities(points)

    for j in range(iterations):
        radius -= 1
        if uniform == 0:
            random.shuffle(points)
        for i in range(lenPoints):
            if uniform == 0:
                chosenIndex = np.random.choice(range(lenPoints))  # returns a random number.
            elif uniform == 1:
                chosenIndex = np.random.choice(range(lenPoints), p=probability)
            elif uniform == 2:
                chosenIndex = np.random.choice(range(lenPoints), p=probability)
          
            bmuIndex = euclideanDistMatrix(points[chosenIndex], matrix)
            matrix = moveNeuroAlgorithmMatrix(points[chosenIndex], bmuIndex, matrix, radius)

            iter = j * lenPoints + i
            if iter % lenPoints == 0:
                drowPaintMatrix(points, matrix, title + " Iter: " + str(iter), chosenIndex=chosenIndex, bmuIndex=bmuIndex)

    drowPaintMatrix(points, matrix, title, done=LAST_ITERATION)


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
    
    xCircle = (LOWER_BOUND + UPPER_BOUND)/2
    yCircle = xCircle
    radiusCircle = math.sqrt(LOWER_BOUND + UPPER_BOUND)
    circle1 = plt.Circle((xCircle, yCircle), radiusCircle, color=CIRCLE_COLOR, fill=False)
    ax = plt.gca()
    ax.add_patch(circle1)
    
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


# def generateCircle(center, numPoints, radius):
#     arc = (2 * math.pi) / numPoints # what is the angle between two of the points
#     points = []

#     for p in range(numPoints):
#         px = (0 * math.cos(arc * p)) - (radius * math.sin(arc * p))
#         py = (radius * math.cos(arc * p)) + (0 * math.sin(arc * p))
#         px += center[0]
#         py += center[1]
#         points.append(Point(px, py))

#     return points


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
    # create neurons
    neurons = generateLine(set=NEURONS_SET, lowerBound=lowerBound, upperBound=upperBound)

    # create points
    points = generateSquare(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)

    # 10 iterations
    neurons = algorithm(points, neurons, iterations=10, title="Line Topology - Uniform Data & Neurons Distribution")
    drowPaintNeurons(points, neurons, "Line Topology - Uniform Data & Neurons Distribution", done=LAST_ITERATION)

    # 40 iterations
    neurons = generateLine(set=NEURONS_SET, lowerBound=0, upperBound=1)
    points = generateSquare(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    neurons = algorithm(points, neurons, iterations=40, title="Line Topology - Uniform Data & Neurons Distribution")
    drowPaintNeurons(points, neurons, "Line Topology - Uniform Data & Neurons Distribution", done=LAST_ITERATION)
    # ------------------------------------------------------- 10x10 Topology -------------------------------------------------------
    # 10 iterations
    points = generateSquare(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    matrix = generateMatrix(lowerBound=lowerBound, upperBound=upperBound, set=NEURONS_SET)

    algorithmMatrix(points, matrix, iterations=10, title="10x10 Topology - Uniform Data & Neurons Distribution")

    # 40 iterations
    points = generateSquare(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    matrix = generateMatrix(lowerBound=lowerBound, upperBound=upperBound, set=NEURONS_SET)

    algorithmMatrix(points, matrix, iterations=40, title="10x10 Topology - Uniform Data & Neurons Distribution")

    # Part A.2
    # ------------------------------------------------------ Non-uniform Dat -------------------------------------------------------
    # ------------------------------------------------------- Line Topology --------------------------------------------------------
    # 10 iterations
    neurons = generateLine(set=NEURONS_SET, lowerBound=0, upperBound=1)
    points = generateSquare(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    
    neurons = algorithm(points, neurons, uniform=1, iterations=10, title="Line Topology - Non-uniform Data & Neurons Distribution")
    drowPaintNeurons(points, neurons, "Line Topology - Non-uniform Data & Neurons Distribution", done=LAST_ITERATION)

    # 40 iterations
    neurons = generateLine(set=NEURONS_SET, lowerBound=0, upperBound=1)
    points = generateSquare(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    
    neurons = algorithm(points, neurons, uniform=1, iterations=40, title="Line Topology - Non-uniform Data & Neurons Distribution")
    drowPaintNeurons(points, neurons, "Line Topology - Non-uniform Data & Neurons Distribution", done=LAST_ITERATION)
    # ------------------------------------------------------- 10x10 Topology -------------------------------------------------------
    # 10 iterations
    points = generateSquare(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    matrix = generateMatrix(lowerBound=lowerBound, upperBound=upperBound, set=NEURONS_SET)

    algorithmMatrix(points, matrix, uniform=2, iterations=10, title="10x10 Topology - Non-uniform Data & Neurons Distribution")

    # 40 iterations
    points = generateSquare(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    matrix = generateMatrix(lowerBound=lowerBound, upperBound=upperBound, set=NEURONS_SET)

    algorithmMatrix(points, matrix, uniform=2, iterations=40, title="10x10 Topology - Non-uniform Data & Neurons Distribution")

    # Part A.3
    # ------------------------------------------------------- Circle Topology ------------------------------------------------------
    # 10 iterations
    neurons = generateLine(set=NEURONS_SMALL_SET, lowerBound=lowerBound, upperBound=upperBound)
    points =  generateCircle(math.sqrt(LOWER_RADIUS), math.sqrt(UPPER_RADIUS), set=DATA_SET)

    neurons = algorithm(points, neurons, iterations=10, title="Circle Topology - Uniform Data & Neurons Distribution", border=1, circleShape=1)
    drowPaintNeurons(points, neurons, "Circle Topology - Uniform Data & Neurons Distribution", border=1, circleShape=1, done=LAST_ITERATION)

    # 40 iterations
    neurons = generateLine(set=NEURONS_SMALL_SET, lowerBound=lowerBound, upperBound=upperBound)
    points =  generateCircle(math.sqrt(LOWER_RADIUS), math.sqrt(UPPER_RADIUS), set=DATA_SET)

    neurons = algorithm(points, neurons, iterations=40, title="Circle Topology - Uniform Data & Neurons Distribution", border=1, circleShape=1)
    drowPaintNeurons(points, neurons, "Circle Topology - Uniform Data & Neurons Distribution", border=1, circleShape=1, done=LAST_ITERATION)

if __name__ == '__main__':
    main()