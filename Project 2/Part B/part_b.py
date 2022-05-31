import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

NEURONS_SET = 225
NEURONS_SMALL_SET = 30
DATA_SET = 300
LAST_ITERATION = 1
DEFAULT_ITERATIONS = 10
MAXSIZE = sys.maxsize
LOWER_BOUND = 0
UPPER_BOUND = 1

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


def algorithmMatrix(points, matrix, iterations=DEFAULT_ITERATIONS, cutOffFinger=0, title=""):
    """
    Function to activate the Kohonen algorithm for matrix
    :param points: Array of points.
    :param matrix: Matrix of neurons.
    :param iterations: Number of train iterations.
    :param cutOffFinger: cutOffFinger = 0 -> draw the hand | cutOffFinger = 1 -> draw the hand with cut off a finger.
    :param title: Title of the task.
    :return: None
    """
    lenPoints = len(points)
    radius = len(matrix[0]) + 1


    for j in range(iterations):
        radius -= 1
        random.shuffle(points)
        for i in range(lenPoints):
            chosenIndex = np.random.choice(range(lenPoints)) 
            bmuIndex = euclideanDistMatrix(points[chosenIndex], matrix)
            matrix = moveNeuroAlgorithmMatrix(points[chosenIndex], bmuIndex, matrix, radius)

            iter = j * lenPoints + i
            if iter % lenPoints == 0:
                drowPaintMatrix(points, matrix, title + " Iter: " + str(iter), chosenIndex=chosenIndex, bmuIndex=bmuIndex, cutOffFinger=0)
    
    if cutOffFinger == 1:
        lastIter = iterations*lenPoints
        for j in range(iterations):
            radius -= 1
            random.shuffle(points)
            for i in range(lenPoints):
                chosenIndex = randomPointInside(lenPoints, points)
                bmuIndex = euclideanDistMatrix(points[chosenIndex], matrix)
                matrix = moveNeuroAlgorithmMatrix(points[chosenIndex], bmuIndex, matrix, radius)

                iter = j * lenPoints + i + lastIter
                if iter % lenPoints == 0:
                    drowPaintMatrix(points, matrix, title + " Iter: " + str(iter), chosenIndex=chosenIndex, bmuIndex=bmuIndex, cutOffFinger=cutOffFinger)

    drowPaintMatrix(points, matrix, title, cutOffFinger=cutOffFinger, done=LAST_ITERATION)


def drowPaintMatrix(points, matrix, title, chosenIndex=-1, bmuIndex=Index(-1, -1), cutOffFinger=0, done=0):
    """
    Function to draw the points and matrix.
    :param points: Array of points.
    :param matrix: Matrix of neurons.
    :param title: Title of the task.
    :param chosenIndex: The chosen point.
    :param bmuIndex: The Best Matching Unit.
    :param cutOffFinger: cutOffFinger = 0 -> draw the hand | cutOffFinger = 1 -> draw the hand with cut off a finger.
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

    if cutOffFinger == 0:
        polygon1 = plt.Polygon([(0,0), (0,0.5), (0.125,1), (0.25,0.5), (0.375,1), (0.5,0.5), (0.625,1), (0.75,0.5), (0.875,1), (1,0.5), (1,0),], color=CIRCLE_COLOR, fill=False)
        ax = plt.gca()
        ax.add_patch(polygon1)
    else:
        polygon1 = plt.Polygon([(0,0), (0,0.5), (0.125,1), (0.25,0.5), (0.5,0.5), (0.625,1), (0.75,0.5), (0.875,1), (1,0.5), (1,0),], color=CIRCLE_COLOR, fill=False)
        ax = plt.gca()
        ax.add_patch(polygon1)

    plt.suptitle(title)
    
    for i in range(len(neurons_x)):
        plt.plot(neurons_x[i], neurons_y[i], NEURON_COLOR)
    
    if done == LAST_ITERATION:
        plt.show()
    else:
        plt.legend(loc="upper left")
        plt.draw()
        plt.pause(0.01)
        plt.clf()

# --------------------------------------------------------- Help Function --------------------------------------------------------
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


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)


def isInside(x1, y1, x2, y2, x3, y3, x, y):
    # Calculate area of triangle ABC
    A = area(x1, y1, x2, y2, x3, y3)
 
    # Calculate area of triangle PBC
    A1 = area(x, y, x2, y2, x3, y3)
     
    # Calculate area of triangle PAC
    A2 = area(x1, y1, x, y, x3, y3)
     
    # Calculate area of triangle PAB
    A3 = area(x1, y1, x2, y2, x, y)
     
    # Check if sum of A1, A2 and A3
    # is same as A
    if(A == A1 + A2 + A3):
        return True
    else:
        return False


def randomPointInside(lenPoints, points):
    while True:
        chosenIndex = np.random.choice(range(lenPoints)) 
        x, y = points[chosenIndex].x, points[chosenIndex].y
        inside = isInside(0.25,0.5, 0.375,1, 0.5,0.5, x, y)
        if inside==False:
            return chosenIndex


def pointOnTriangle(pt1, pt2, pt3):
    """
    Random point on the triangle with vertices pt1, pt2 and pt3.
    """
    x, y = random.random(), random.random()
    q = abs(x - y)
    s, t, u = q, 0.5 * (x + y - q), 1 - 0.5 * (q + x + y)
    return (
        s * pt1[0] + t * pt2[0] + u * pt3[0],
        s * pt1[1] + t * pt2[1] + u * pt3[1],
    )


def generateHand(set, lowerBound , upperBound):
    """
    Function generate the data points.
    :param set: The number of points.
    :param lowerBound: The lower bound for a lower parameter in random.uniform function.
    :param upperBound: The upper bound for a high parameter in random.uniform function.
    :return: The data points
    """
    handData = []
    palm = int(set/2)
    fingers = int(set/8)

    # the data set is {(x,y) |  0 <= x <= 1, 0<=y<=1}
    for _ in range(palm):
        handData.append(Point(random.uniform(lowerBound, upperBound), random.uniform(lowerBound, upperBound/2)))
        
    # first finger
    pt1 = (lowerBound, upperBound/2)
    pt2 = (upperBound/8, upperBound)
    pt3 = (upperBound/4, upperBound/2)
    for _ in range(fingers):
        point = pointOnTriangle(pt1, pt2, pt3)
        handData.append(Point(point[0], point[1]))

    # second finger
    pt1 = (upperBound/4, upperBound/2)
    pt2 = (upperBound/4+upperBound/8, upperBound)
    pt3 = (upperBound/2, upperBound/2)
    for _ in range(fingers):
        point = pointOnTriangle(pt1, pt2, pt3)
        handData.append(Point(point[0], point[1]))

    # third finger
    pt1 = (upperBound/2, upperBound/2)
    pt2 = (upperBound/2+upperBound/8, upperBound)
    pt3 = (upperBound-upperBound/4, upperBound/2)
    for _ in range(fingers):
        point = pointOnTriangle(pt1, pt2, pt3)
        handData.append(Point(point[0], point[1]))

    # fourth finger
    pt1 = (upperBound-upperBound/4, upperBound/2)
    pt2 = (upperBound-upperBound/8, upperBound)
    pt3 = (upperBound, upperBound/2)
    for _ in range(fingers):
        point = pointOnTriangle(pt1, pt2, pt3)
        handData.append(Point(point[0], point[1]))

    return handData


def main():
    lowerBound = LOWER_BOUND
    upperBound = UPPER_BOUND
    # Part B.1
    # ------------------------------------------------------- 15x15 Topology -------------------------------------------------------
    # 10 iterations
    points = generateHand(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    matrix = generateMatrix(lowerBound=lowerBound, upperBound=upperBound, set=NEURONS_SET)

    algorithmMatrix(points, matrix, iterations=10, title="15x15 Topology - Uniform Data & Neurons Distribution")

    # 40 iterations
    points = generateHand(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    matrix = generateMatrix(lowerBound=lowerBound, upperBound=upperBound, set=NEURONS_SET)

    algorithmMatrix(points, matrix, iterations=40, title="15x15 Topology - Uniform Data & Neurons Distribution")

    # Part B.2
    # ------------------------------------------------------- 15x15 Topology -------------------------------------------------------
    # 10 iterations
    points = generateHand(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    matrix = generateMatrix(lowerBound=lowerBound, upperBound=upperBound, set=NEURONS_SET)

    algorithmMatrix(points, matrix, iterations=10, cutOffFinger=1, title="15x15 Topology - Uniform Data & Neurons Distribution")

    # 40 iterations
    points = generateHand(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    matrix = generateMatrix(lowerBound=lowerBound, upperBound=upperBound, set=NEURONS_SET)

    algorithmMatrix(points, matrix, iterations=40, cutOffFinger=1, title="15x15 Topology - Uniform Data & Neurons Distribution")



if __name__ == '__main__':
    main()