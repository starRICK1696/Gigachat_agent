"""QUBO Evaluation module"""


from npqtools.problems.QuboSalesman import QUBOSalesman
from npqtools.problems.QuboClique import QUBOClique
from npqtools.problems.QuboKnapsack import QUBOKnapsack
from npqtools.problems.QuboTiling2Dim import QUBOTiling2Dim
from npqtools.problems.QuboProductionScheduling import QUBOProductionScheduling
from npqtools.problems.QuboMultiKnapsack import QUBOMultiKnapsack
from npqtools.problems.QuboMaxWeightClique import QUBOMaxWeightClique
import numpy as np


class BasicEvaluator:
    def __init__(self, matrix=None):
        self.matrix = matrix

    def evaluation_function(self, **kwargs):
        exit("Error : evaluation function is not implemented. ")

    def evaluate(self, display=False, num_reads=100, arglist=None):
        if arglist is None:
            arglist = []

        if self.matrix is None:
            exit("Error : Input data is incorrect")

        eval = self.evaluation_function(arglist)

        eval.find_min_energy(num_reads)

        return eval.solution


class SalesmanEvaluator(BasicEvaluator):
    def __init__(self, matrix=None, from_coordinates=False):
        self.from_coordinates = from_coordinates
        super().__init__(matrix)

    def evaluation_function(self, **kwargs):
        if self.from_coordinates:
            return QUBOSalesman(coord_matrix=self.matrix)
        return QUBOSalesman(adjacency_matrix=self.matrix)



class CliqueEvaluator(BasicEvaluator):
    def __init__(self, matrix=None):
        super().__init__(matrix)

    def evaluation_function(self, **kwargs):
        return QUBOClique(self.matrix)
    

class KnapsackEvaluator(BasicEvaluator):
    def __init__(self, matrix=None):
        super().__init__(matrix)

    def evaluation_function(self, capacity, **kwargs):
        return QUBOKnapsack(self.matrix, capacity)


class Tiling2DimEvaluator(BasicEvaluator):
    def __init__(self, matrix=None):
        super().__init__(matrix)

    def tester_function(self, width, length, **kwargs):
        return QUBOTiling2Dim(self.matrix, width, length)


class ProductionSchedulingEvaluator(BasicEvaluator):
    def __init__(self, matrix=None):
        super().__init__(matrix)

    def tester_function(self, setup_times, job_values, **kwargs):
        return QUBOProductionScheduling(self.matrix, setup_times, job_values)


class MultiKnapEvaluator(BasicEvaluator):
    def __init__(self, matrix=None):
        super().__init__(matrix)

    def tester_function(self, capabilities, **kwargs):
        return QUBOMultiKnapsack(self.matrix, capabilities)


class MaxWeightCliqueEvaluator(BasicEvaluator):
    def __init__(self, matrix=None):
        super().__init__(matrix)

    def tester_function(self, **kwargs): 
        return QUBOMaxWeightClique(self.matrix)




