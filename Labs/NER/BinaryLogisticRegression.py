from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.01  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 100 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + math.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """

        # REPLACE THE COMMAND BELOW WITH YOUR CODE

        return self.sigmoid(np.matmul(np.transpose(self.theta), self.x[datapoint]))


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        # YOUR CODE HERE
        sum = 0
        for k in range(0,self.FEATURES):
            for i in range(0,self.DATAPOINTS):
                z = np.matmul(np.transpose(self.theta),self.x[i])
                h = self.sigmoid(z)
                sub = h - self.y[i]
                sum += self.x[i][k]*sub 
            sum = sum/self.DATAPOINTS
            self.gradient[k] = sum
        return self.gradient

    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        
        # YOUR CODE HERE
        sum = 0
        for k in range(0,self.FEATURES):
            for i in range(0,len(minibatch)):
                z = np.matmul(np.transpose(self.theta),self.x[minibatch[i]])
                h = self.sigmoid(z)
                sub = h - self.y[minibatch[i]]
                sum += self.x[minibatch[i]][k]*sub 
            sum = sum/len(minibatch)
            self.gradient[k] = sum
        return self.gradient        


    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        # YOUR CODE HERE
        sum = 0
        for k in range(0,self.FEATURES):
            z = np.matmul(np.transpose(self.theta),self.x[datapoint])
            h = self.sigmoid(z)
            sub = h - self.y[datapoint] 
            self.gradient[k] = self.x[datapoint][k]*sub
        return self.gradient
            
    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        converged = False
        ite = 0
        while not converged:
            datapoint = random.randint(0, self.DATAPOINTS)
            grad = self.compute_gradient(datapoint)   
            for k in range(0, self.FEATURES):
                self.theta[k] = self.theta[k] - self.LEARNING_RATE*grad[k]
            conv = sum([(i*i) for i in grad])
            # if abs(conv) <= self.CONVERGENCE_MARGIN:
                # converged = True
            if ite == self.MAX_ITERATIONS:
                converged = True
            self.update_plot(ite, conv)
            ite = ite + 1


    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        if self.DATAPOINTS < self.MINIBATCH_SIZE:
            batch_size = self.DATAPOINTS
        else:
            batch_size = self.MINIBATCH_SIZE
        
        converged = False
        iter = 0
        while not converged:
            minibatch = random.sample(range(self.DATAPOINTS+1), batch_size)
            grad = self.compute_gradient_minibatch(minibatch)   
            for k in range(0, self.FEATURES):
                self.theta[k] = self.theta[k] - self.LEARNING_RATE*grad[k]
            
            conv = sum([(i*i) for i in grad])
            self.update_plot(iter, conv)
            
            if abs(conv) <= self.CONVERGENCE_MARGIN:
                converged = True
            
            iter += 1
            if iter == self.MAX_ITERATIONS:
                converged = True        


    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        i = 0
        converged = False
        while not converged:
            self.update_plot(i, conv)
            grad = self.compute_gradient_for_all()
            for k in range(0, self.FEATURES):
                self.theta[k] = self.theta[k] - self.LEARNING_RATE*grad[k]
            
            conv = sum([(i*i) for i in grad])            
            i += 1
            if abs(conv) <= self.CONVERGENCE_MARGIN:
                converged = True   
                


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))
        result = []

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            result.append(predicted)
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))
        pred_result = np.array(result)
        #calculate metrics
        t_p = 0
        f_p = 0
        t_n = 0
        f_n = 0
        for i in range(self.DATAPOINTS):
            if self.y[i] == pred_result[i]:
                if pred_result[i] == 0:
                    f_p += 1
                if pred_result[i] == 1:
                    t_p +=1
            else:
                if pred_result[i] == 1:
                    t_n +=1
                else:
                    f_n +=1
        print('Accuracy: {}'.format((t_p+f_p)/self.DATAPOINTS))
        print('For class NAME')
        print('Precision: {}'.format(t_p/(t_p+f_p)))
        print('Recall: {}'.format(t_p/(t_p+f_n)))
        print('For class NOT NAME')
        print('Precision: {}'.format(f_p/(f_p+t_p)))
        print('Recall: {}'.format(f_p/(f_p+t_n)))



    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=np.random.rand(3,1), linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()
