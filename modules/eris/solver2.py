import numpy as np

import _eris

from eris.problem2 import Problem
from eris.transformations import (
    quaternion_from_matrix,
    quaternion_matrix,
    inverse_matrix,
    random_quaternion,
    random_vector,
)

class Solver:
    def __init__(self):
        pass

    def _solve(self, problem: Problem, x=None):
        """
        Solve the given problem (starting from a 'random' initial guess if the optional argument is not provided).
        """
        if x is not None:
            q0, t0 = x
        else:
            q0 = np.roll(random_quaternion(), 1)
            t0 = random_vector(3)

        campoints = problem.campoints
        campoints_true = problem.campoints_true
        robposes = problem.robposes

        solver = _eris.Solver(q0, t0)

        for pose in robposes:
            for i, points in enumerate(campoints):
                Ti = pose
                qi = np.roll(quaternion_from_matrix(Ti), 1)
                ti = Ti[:3, 3]
                pis = points
                for i, pi in enumerate(pis):
                    pj = campoints_true[i]
                    solver.add_residual_block(qi, ti, pi, pj)

        qopt, topt = solver.solve()
        Xopt = quaternion_matrix(np.roll(qopt, -1))
        Xopt[:3, 3] = topt

        summary = _eris.summary_to_dict(solver.summary())

        return Xopt, summary

    def calibrate_eye_in_hand(self, problem, x=None):
        return self._solve(problem, x)

    def calibrate_eye_to_hand(self, problem, x=None):
        problem.robposes = [inverse_matrix(robpose) for robpose in problem.robposes]
        return self._solve(problem, x)