import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

NEURONS_SET = 225
DATA_SET = 600
LAST_ITERATION = 1
DEFAULT_ITERATIONS = 10
MAXSIZE = sys.maxsize
LOWER_BOUND = 0
UPPER_BOUND = 1
LIST_PRINT = [100, 10000, 20000, 30000, 40000, 50000, 55000, 60000]

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
    def __init__(self, set=NEURONS_SET, lowerBound=LOWER_BOUND, upperBound=UPPER_BOUND, learning_rate=.5, neighborhood_distance=3):
        self.neurons = generateMatrix(lowerBound=lowerBound, upperBound=upperBound, set=set)

        self.eps = learning_rate   # initial learning speed
        self.de = neighborhood_distance   # initial neighborhood distance
        self.ste = 0    # inital number of carried out steps
    
    def phi2(self, ix, iy, kx, ky, d):  # proximity function for matrix
        return np.exp(-((ix-kx)**2+(iy-ky)**2)/(d**2))  # Gaussian

    def train(self, data, title, rounds=100, points=300, cutOffFinger=0):
        """
        Function to activate the Kohonen algorithm 
        :param data: the data to be trained on.
        :param title: Title of the task.
        :param rounds: number of rounds.
        :param points: number of points in each round.
        :param cutOffFinger: cutOffFinger = 0 -> draw the hand | cutOffFinger = 1 -> draw the hand with cut off a finger.
        :return:
        """
        lenData = len(data)
        rows = len(self.neurons)
        cols = len(self.neurons[0])
        if cutOffFinger == 1:
            self.eps = .5
            self.de = 3
        
        self.drow(data, title, chosenIndex=-1, bmuIndex=Index(-1, -1), cutOffFinger=cutOffFinger)
        for _ in range(rounds):   # rounds
            self.eps = self.eps*.97      
            self.de = self.de*.98         
            for _ in range(points):    # repeat for rep points
                self.ste = self.ste+1

                if cutOffFinger == 1:
                    chosenIndex = randomPointInside(lenData, data)
                else:
                    chosenIndex = np.random.choice(range(lenData)) 
                bmuIndex = self.euclideanDist(data[chosenIndex])
                
                ind_i=bmuIndex.x
                ind_j=bmuIndex.y    
        
                for j in range(rows): 
                    for i in range(cols):
                        self.neurons[i][j].point.x += self.eps*self.phi2(ind_i,ind_j,i,j,self.de)*(data[chosenIndex].x - self.neurons[i][j].point.x) 
                        self.neurons[i][j].point.y += self.eps*self.phi2(ind_i,ind_j,i,j,self.de)*(data[chosenIndex].y - self.neurons[i][j].point.y)
                
                if  self.ste in LIST_PRINT:
                    self.drow(data, title + " Iter: " + str(self.ste), chosenIndex=chosenIndex, bmuIndex=bmuIndex, cutOffFinger=cutOffFinger)

        self.drow(data, title + " Iter: " + str(self.ste), chosenIndex=-1, bmuIndex=Index(-1, -1), cutOffFinger=cutOffFinger, done=LAST_ITERATION)
    

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
        index = Index()
        for i in range(len(self.neurons[0])):
            for j in range(len(self.neurons[0])):
                if self.neurons[i][j].point.chosen == 0:
                    distance = math.sqrt((data.x - self.neurons[i][j].point.x) ** 2 + (data.y - self.neurons[i][j].point.y) ** 2)
                    if distance < minimum:
                        minimum = distance
                        index = Index(i, j)
        return index


    def drow(self, points, title, chosenIndex=-1, bmuIndex=Index(-1, -1), cutOffFinger=0, done=0):
        """
        Function to draw the points and matrix.
        :param points: Array of points.
        :param title: Title of the task.
        :param chosenIndex: The chosen point.
        :param bmuIndex: The Best Matching Unit.
        :param cutOffFinger: cutOffFinger = 0 -> draw the hand | cutOffFinger = 1 -> draw the hand with cut off a finger.
        :param done: Done = 0 -> draw the matrix and clear | Done = 1 -> last iteration, show the matrix.
        :return: None
        """
        neurons_x = [[] for _ in range(2 * len(self.neurons[0]))]
        neurons_y = [[] for _ in range(2 * len(self.neurons[0]))]
    
        for i in range(len(points)):
            if done == 0 and chosenIndex == i:
                plt.scatter(points[i].x, points[i].y, color=CHOSEN_POINT_COLOR, label='Chosen data point')
            else:
                plt.scatter(points[i].x, points[i].y, color=POINT_COLOR)
    
        index = 0
        for i in range(len(self.neurons[0])):
            for j in range(len(self.neurons[0])):
                if done == 0 and bmuIndex.x == i and bmuIndex.y == j:
                    plt.scatter(self.neurons[i][j].point.x, self.neurons[i][j].point.y, color=BMU_COLOR, label='BMU')
                else:
                    plt.scatter(self.neurons[i][j].point.x, self.neurons[i][j].point.y, color=NEURON_COLOR)
                neurons_x[index].append(self.neurons[i][j].point.x)
                neurons_y[index].append(self.neurons[i][j].point.y)
            index += 1
    
        for i in range(len(self.neurons[0])):
            for j in range(len(self.neurons[0])):
                neurons_x[index].append(self.neurons[j][i].point.x)
                neurons_y[index].append(self.neurons[j][i].point.y)
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
            plt.show()
            # plt.draw()
            # plt.pause(0.01)
            # plt.clf()
    

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

# --------------------------------------------------------- Help Function --------------------------------------------------------
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


def generateHand(set, lowerBound, upperBound):
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
    points = generateHand(set=DATA_SET, lowerBound=lowerBound, upperBound=upperBound)
    
    # 30000 iterations
    somSquare = KohonenAlgorithm(set=NEURONS_SET, lowerBound=lowerBound, upperBound=upperBound)
    somSquare.train(points, cutOffFinger=0, rounds=100, points=300, title="15x15 Topology - Uniform Data")

    # Part B.2
    # ------------------------------------------------------- 15x15 Topology -------------------------------------------------------
    # 30000 iterations
    somSquare.train(points, cutOffFinger=1, rounds=100, points=400, title="15x15 Topology - Uniform Data & after cut")



if __name__ == '__main__':
    main()