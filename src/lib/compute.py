"""QUBO Evaluation module — wrappers around npqtools solvers."""


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

    def _evaluation_function(self, matrix, **kwargs):
        raise NotImplementedError("Evaluation function is not implemented.")

    def _return(self):
        raise NotImplementedError("Return function is not implemented.")

    def evaluate(self, matrix, display=False, num_reads=100, *args, **kwargs):
        self.evaluator = self._evaluation_function(matrix, *args, **kwargs)
        self.evaluator.find_min_energy(num_reads)
        return self._return()


class SalesmanEvaluator(BasicEvaluator):
    def _evaluation_function(self, matrix, from_coordinates=False, **kwargs):
        if from_coordinates:
            return QUBOSalesman(coord_matrix=matrix)
        return QUBOSalesman(adjacency_matrix=matrix)

    def _return(self):
        if self.evaluator is None:
            raise RuntimeError("Evaluator has not been run yet.")
        answer = 0
        for i in range(len(self.evaluator.solution)):
            answer += self.evaluator.adjacency_matrix[
                self.evaluator.solution[i]
            ][
                self.evaluator.solution[(i + 1) % len(self.evaluator.solution)]
            ]
        return {"answer": answer, "characteristics": self.evaluator.solution}


class CliqueEvaluator(BasicEvaluator):
    def _evaluation_function(self, matrix, **kwargs):
        return QUBOClique(matrix)

    def _return(self):
        return {
            "answer": len(self.evaluator.solution),
            "characteristics": self.evaluator.solution,
        }


class KnapsackEvaluator(BasicEvaluator):
    def _evaluation_function(self, matrix, capacity, **kwargs):
        return QUBOKnapsack(matrix, capacity)

    def _return(self):
        return {
            "answer": int(-self.evaluator.min_energy),
            "total_weight": int(self.evaluator.sum_weight),
            "characteristics": self.evaluator.solution,
        }


class Tiling2DimEvaluator(BasicEvaluator):
    def _evaluation_function(self, matrix, width, length, banned=None, separations=None, **kwargs):
        return QUBOTiling2Dim(matrix, width, length, banned=banned, separations=separations)

    def _return(self):
        return {
            "answer": sum([sum(row) for row in self.evaluator.solution]),
            "characteristics": self.evaluator.solution,
        }


class ProductionSchedulingEvaluator(BasicEvaluator):
    def _evaluation_function(self, matrix, setup_times, job_values, **kwargs):
        return QUBOProductionScheduling(matrix, setup_times, job_values)

    def _return(self):
        return {
            "answer": self.evaluator.min_energy + self.evaluator.delta,
            "characteristics": self.evaluator.solution,
        }


class MultiKnapEvaluator(BasicEvaluator):
    def _evaluation_function(self, matrix, capabilities, **kwargs):
        return QUBOMultiKnapsack(matrix, capabilities)

    def _return(self):
        return {
            "answer": -self.evaluator.min_energy,
            "characteristics": self.evaluator.solution,
        }


class MaxWeightCliqueEvaluator(BasicEvaluator):
    def _evaluation_function(self, matrix, **kwargs):
        return QUBOMaxWeightClique(matrix)

    def _return(self):
        sum_edge = 0
        for i in range(len(self.evaluator.solution)):
            for j in range(i + 1, len(self.evaluator.solution)):
                sum_edge += self.evaluator.adjacency_matrix[
                    self.evaluator.solution[i]
                ][
                    self.evaluator.solution[j]
                ]
        return {"answer": sum_edge, "characteristics": self.evaluator.solution}
