import sys
import math
import random
import matplotlib.pyplot as plt

SET_NEURONS = 100
NUMBER_NEURONS_ARRAY = 10
LAST_ITERATION = 1
MAXSIZE = sys.maxsize
NEURON_COLOR = '#1c34d1'
POINT_COLOR = '#1cd1cb'
CIRCLE_COLOR = '#d11c58'

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


def minDistance(point, neurons):
    """
    Function to find the minimum distance
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
    neurons[index].x += (point.x - neurons[index].x) / 2
    neurons[index].y += (point.y - neurons[index].y) / 2
    neurons[index].chosen = 1
 
    for i in range(1, radius + 1):
        i_right = index + i
        i_left = index - i
        delta = 1 / (2 ** (i + 1))
        
        if i_right < len(neurons):
            neurons[i_right].x += (point.x - neurons[i_right].x) * delta
            neurons[i_right].y += (point.y - neurons[i_right].y) * delta
            neurons[i_right].chosen = 1
        
        if i_left >= 0:
            neurons[i_left].x += (point.x - neurons[i_left].x) * delta
            neurons[i_left].y += (point.y - neurons[i_left].y) * delta
            neurons[i_left].chosen = 1

    return updateConscience(index, neurons)


def drowPaintNeurons(points, neurons, title, circleShape=0, done=0):
    """
    Function to draw the points and neurons.
    :param points: Array of points
    :param neurons: Array of neurons
    :param title: Title of the task.
    :param circleShape: circleShape = 0 -> line topology | circleShape = 1 -> circle topology.
    :param done: Done = 0 -> draw the board and clear | Done = 1 -> last iteration, show the board.
    :return: None
    """
    neurons_x = []
    neurons_y = []

    for i in range(len(points)):
        plt.scatter(points[i].x, points[i].y, color=POINT_COLOR)
    
    for i in range(len(neurons)):
        neurons_x.append(neurons[i].x)
        neurons_y.append(neurons[i].y)
        plt.scatter(neurons[i].x, neurons[i].y, color=NEURON_COLOR)
    if circleShape == 1:
        neurons_x.append(neurons_x[0]), neurons_y.append(neurons_y[0])

    circle = plt.Circle((0,0), 2, color=CIRCLE_COLOR, fill=False)
    ax = plt.gca()
    ax.add_patch(circle)

    plt.suptitle(title)
    plt.plot(neurons_x, neurons_y)

    if done == LAST_ITERATION:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.01)
        plt.clf()


def algorithm(points, neurons, title="", circleShape=0):
    """
    Function to activate the Kohonen algorithm 
    :param points: Array of points
    :param neurons: Array of neurons
    :param title: Title of the task
    :param circleShape: circleShape = 0 -> line topology | circleShape = 1 -> circle topology.
    :return: The updated neurons locations according to the given radius.
    """
    drowPaintNeurons(points, neurons, title, circleShape)

    size = int(len(points)/len(neurons)*2) + 1
    radius = int(len(neurons)/2)

    for i in range(len(points)):
        if i % size == 0 and i != 0:
            radius -= 1
        new_neurons = moveNeuronAlgorithm(points[i], minDistance(points[i], neurons), neurons, radius)
        if i % 50 == 0:
            drowPaintNeurons(points, new_neurons, title + " Iter: " + str(i), circleShape)

    return new_neurons


def createPoints(x, radius1, radius2=0):
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

# -------------------------------------------------------------- Matrix -------------------------------------------------
def createTwoDimensionalArray(neurons):
    """
    :param neurons: 10x10 neurons (Total: 100)
    :return: Neurons arranged in a 10x10 topology.
    """
    matrix = [[Node() for i in range(NUMBER_NEURONS_ARRAY)] for j in range(NUMBER_NEURONS_ARRAY)]

    "Corners"
    matrix[0][0] = Node(neurons[0][0], Index(0, 0), [Index(0, 1), Index(1, 0)])
    matrix[0][NUMBER_NEURONS_ARRAY-1] = Node(neurons[0][NUMBER_NEURONS_ARRAY-1], Index(0, 4), [Index(0, 3), Index(1, 4)])
    matrix[NUMBER_NEURONS_ARRAY-1][0] = Node(neurons[NUMBER_NEURONS_ARRAY-1][0], Index(4, 0), [Index(3, 0), Index(4, 1)])
    matrix[NUMBER_NEURONS_ARRAY-1][NUMBER_NEURONS_ARRAY-1] = Node(neurons[NUMBER_NEURONS_ARRAY-1][NUMBER_NEURONS_ARRAY-1], Index(4, 4), [Index(3, 4), Index(4, 3)])

    for i in range(1, NUMBER_NEURONS_ARRAY-1):
        "Edges"
        matrix[0][i] = Node(neurons[0][i], Index(0, i), [Index(0, i - 1), Index(1, i), Index(0, i + 1)])
        matrix[i][0] = Node(neurons[i][0], Index(i, 0), [Index(i - 1, 0), Index(i, 1), Index(i + 1, 0)])
        matrix[NUMBER_NEURONS_ARRAY-1][i] = Node(neurons[NUMBER_NEURONS_ARRAY-1][i], Index(4, i), [Index(4, i - 1), Index(3, i), Index(4, i + 1)])
        matrix[i][NUMBER_NEURONS_ARRAY-1] = Node(neurons[i][NUMBER_NEURONS_ARRAY-1], Index(i, 4), [Index(i - 1, 4), Index(i, 3), Index(i + 1, 4)])

        "General Case"
        for j in range(1, NUMBER_NEURONS_ARRAY-1):
            matrix[i][j] = Node(neurons[i][j], Index(i, j), [Index(i, j-1), Index(i-1, j), Index(i, j+1), Index(i+1, j)])

    return matrix


def minDistanceMatrix(point, matrix):
    """
    Function to find the minimum distance for task C
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
    matrix[index.x][index.y].point.change = 1

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
                if node.point.change == 0:
                    matrix[locate.x][locate.y].point.x += (point.x - node.point.x) * delta
                    matrix[locate.x][locate.y].point.y += (point.y - node.point.y) * delta
                    matrix[locate.x][locate.y].point.change = 1
                    for k in range(len(node.adjacent)):
                        queue2.append(node.adjacent[k])

        if counter % 2 == 0:
            while len(queue2) > 0:
                locate = queue2.pop(0)
                node = matrix[locate.x][locate.y]
                if node.point.change == 0:
                    matrix[locate.x][locate.y].point.x += (point.x - node.point.x) * delta
                    matrix[locate.x][locate.y].point.y += (point.y - node.point.y) * delta
                    matrix[locate.x][locate.y].point.change = 1
                    for k in range(len(node.adjacent)):
                        queue1.append(node.adjacent[k])

    return updateConscienceMatrix(index, matrix)


def algorithmMatrix(points, matrix, title=""):
    """
    Function to activate the Kohonen algorithm for matrix
    :param points: Array of points
    :param matrix: Matrix of neurons
    :param title: Title of the task
    :return: None
    """
    drowPaintMatrix(points, matrix, title)
    radius = len(matrix[0]) + 1

    for j in range(len(matrix[0])):
        radius -= 1
        random.shuffle(points)
        for i in range(len(points)):
            new_matrix = moveNeuroAlgorithmMatrix(points[i], minDistanceMatrix(points[i], matrix), matrix, radius)
            if i % 50 == 0:
                drowPaintMatrix(points, new_matrix, title + " Iter: " + str(i))

    drowPaintMatrix(points, new_matrix, title, done=LAST_ITERATION)


def drowPaintMatrix(points, matrix, title, done=0):
    """
    Function to draw the points and matrix.
    :param points: Array of points
    :param matrix: Matrix of neurons
    :param title: Title of the task
    :param done: Done = 0 -> draw the matrix and clear | Done = 1 -> last iteration, show the matrix.
    :return: None
    """
    neurons_x = [[] for i in range(2 * len(matrix[0]))]
    neurons_y = [[] for i in range(2 * len(matrix[0]))]
    
    for i in range(len(points)):
        plt.scatter(points[i].x, points[i].y, color=POINT_COLOR)
    index = 0
    
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
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
    
    circle = plt.Circle((0, 0), 2, color=CIRCLE_COLOR, fill=False)
    ax = plt.gca()
    ax.add_patch(circle)
    
    if done == LAST_ITERATION:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.01)
        plt.clf()


def main():
    radius = 2
    # ------------------------------------------------------- Line Topology -------------------------------------------------------
    neurons = []
    points = []
    # create neurons
    for i in range(SET_NEURONS):
        neurons.append(Point(random.uniform(0, 1), random.uniform(0, 1)))

    #  create points
    for i in range(200):
        x = random.uniform(-radius, radius)
        points.append(Point(x, createPoints(x, radius)))

    for i in range(10):
        random.shuffle(points)
        neurons = algorithm(points, neurons, "Line Topology - Uniform Data & Neurons Distribution")

    drowPaintNeurons(points, neurons, "Line Topology - Uniform Data & Neurons Distribution", done=LAST_ITERATION)

    # ------------------------------------------------------- Circle Topology -------------------------------------------------------
    neurons = []
    points = []
    for i in range(30):
        neurons.append(Point(random.uniform(0, 1), random.uniform(0, 1)))

    for i in range(200):
        x = random.uniform(-radius, radius)
        points.append(Point(x, createPoints(x, radius)))

    for i in range(10):
        random.shuffle(points)
        neurons = algorithm(points, neurons, "Circle Topology - Uniform Data & Neurons Distribution", circleShape=1)

    drowPaintNeurons(points, neurons, "Circle Topology - Uniform Data & Neurons Distribution", circleShape=1, done=LAST_ITERATION)

    # ------------------------------------------------------- 10x10 Topology -------------------------------------------------------
    points = []
    for i in range(200):
        x = random.uniform(-radius, radius)
        points.append(Point(x, createPoints(x, radius)))

    neurons = [[Point() for i in range(NUMBER_NEURONS_ARRAY)] for j in range(NUMBER_NEURONS_ARRAY)]
    for i in range(NUMBER_NEURONS_ARRAY):
        for j in range(NUMBER_NEURONS_ARRAY):
            neurons[i][j] = Point(random.uniform(0, 1), random.uniform(0, 1))
    board = createTwoDimensionalArray(neurons)
    
    algorithmMatrix(points, board, "10x10 Topology - Uniform Data & Neurons Distribution")

if __name__ == '__main__':
    main()