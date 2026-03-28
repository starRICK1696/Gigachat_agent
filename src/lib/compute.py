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
    def __init__(self):
        self.evaluator = None

    def __evaluation_function(self, matrix, **kwargs):
        raise NotImplementedError("Evaluation function is not implemented.")

    def __return(self):
        raise NotImplementedError("Return function is not implemented.")

    def evaluate(self, matrix, display=False, num_reads=100, *args, **kwargs):
        self.evaluator = self.__evaluation_function(matrix, *args, **kwargs)
        self.evaluator.find_min_energy(num_reads)
        return self.__return()


class SalesmanEvaluator(BasicEvaluator):
    def __evaluation_function(self, matrix, from_coordinates=False, **kwargs):
        if from_coordinates:
            return QUBOSalesman(coord_matrix=matrix)
        return QUBOSalesman(adjacency_matrix=matrix)
    
    def __return(self):
        if self.evaluator is None:
            raise NotImplementedError("Evaluation function is not implemented.")
        answer = 0
        for i in range(len(self.evaluator.solution)):
            answer += self.evaluator.adjacency_matrix[self.evaluator.solution[i]][self.evaluator.solution[(i + 1) % len(self.evaluator.solution)]]
        return {"answer" : answer, "characteristics" : self.evaluator.solution}


class CliqueEvaluator(BasicEvaluator):
    def __evaluation_function(self, matrix, **kwargs):
        return QUBOClique(matrix)
    
    def __return(self):
        return {"answer" : len(self.evaluator.solution), "characteristics" : self.evaluator.solution}


class KnapsackEvaluator(BasicEvaluator):
    def __evaluation_function(self, matrix, capacity, **kwargs):
        return QUBOKnapsack(matrix, capacity)
    
    def __return(self):
        return {"answer" : self.sum_weight, "characteristics" : self.evaluator.solution}


class Tiling2DimEvaluator(BasicEvaluator):
    def __evaluation_function(self, matrix, width, length, **kwargs):
        return QUBOTiling2Dim(matrix, width, length)
    
    def __return(self):
        return {"answer" : sum([sum(row) for row in self.evaluator.solution]), "characteristics" : self.evaluator.solution}


class ProductionSchedulingEvaluator(BasicEvaluator):
    def __evaluation_function(self, matrix, setup_times, job_values, **kwargs):
        return QUBOProductionScheduling(matrix, setup_times, job_values)
    
    def __return(self):
        return {"answer" : "some loss", "characteristics" : self.evaluator.solution}


class MultiKnapEvaluator(BasicEvaluator):
    def __evaluation_function(self, matrix, capabilities, **kwargs):
        return QUBOMultiKnapsack(matrix, capabilities)
    
    def __return(self):
        return {"answer" : -self.min_energy, "characteristics" : self.evaluator.solution}


class MaxWeightCliqueEvaluator(BasicEvaluator):
    def __evaluation_function(self, matrix, **kwargs):
        return QUBOMaxWeightClique(matrix)
    
    def __return(self):
        sum_edge = 0
        for i in range(len(self.evaluator.solution)):
            for j in range(i + 1, len(self.evaluator.solution)):
                sum_edge += self.evaluator.adjacency_matrix[self.evaluator.solution[i]][self.evaluator.solution[j]]
        return {"answer" : len(self.evaluator.solution), "characteristics" : self.evaluator.solution, "sum_edges" : sum_edge}