# Neural_Networks_5782
Course at Ariel University during the spring semester.
## Project 1
There are four parts to homework:
- Part A: Implement the Adaline learning algorithm and show how it generalizes to develop a decision that works on all the set. Only when x,y <= 100 does <x,y> have value 1.
- Part B: The Adaline learning algorithm from part A is used for the new formula. Only when 4 <= x^2 + y^2 <= 9 does <x,y> have value 1. The algorithm should fail.
- Part C: Instead of adaline, use back-propagation algorithm for the formule: 4 <= x^2 + y^2 <= 9
- Part D: The neurons from the next to last level of Part C are used as input and only an Adaline is output.
## Project 2
There are two parts to homework:
- Part A:  Implement the Kohonen algorithm and use it to fit:
  * A set of  100 neurons in a topology of a line to a disk. (That is, the data set is {(x,y) |  0 <= x <= 1, 0<=y<=1} for  which the distribution is uniform while the Kohonen level is linearly ordered.) Do the same when the topology of the 100 neurons is arranged in a two dimensional array of 10 x 10.
  * Do the same with at least two non-uniform distributions on the disk.  
  * Do the same experiments for fitting a circle of neurons on a "donut" shape i.e. {<x.y> | 2<= x^2 +y^2 <= 4}.  The line of neurons has 30 neurons organized as a circle topology.
- Part B: Reproduce the experiment on the "monkey hand" as described in class. For this part you can use your own code or you may use the Kohonen algorithm in a package (like the SOM package in Matlab).  
