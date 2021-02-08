import numpy as np

import _eris

from eris.sim_problem import Problem
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

    def _solve(self, problem: Problem, x1=None, x2 = None):
        """
        Solve the given problem (starting from a 'random' initial guess if the optional argument is not provided).
        """
        if x1 is not None:
            q0, t0 = x1
        if x2 is not None:
            q1, t1 = x2
        else:
            q0 = np.roll(random_quaternion(), 1)
            t0 = random_vector(3)
            q1 = np.roll(random_quaternion(), 1)
            t1 = random_vector(3)
            

        campoints1 = problem.campoints1
        robposes1 = problem.robposes1
        
        campoints2 = problem.campoints2
        robposes2 = problem.robposes2


        solver = _eris.Solver(q0, t0, q1, t1)

        n = len(robposes)
        for i in range(n):
            for j in range(i + 1, n):
                Ti1 = robposes1[i]
                qi1 = np.roll(quaternion_from_matrix(Ti1), 1)
                ti1 = Ti1[:3, 3]
                pi1s = campoints1[i]

                Tj1 = robposes1[j]
                qj1 = np.roll(quaternion_from_matrix(Tj1), 1)
                tj1 = Tj1[:3, 3]
                pj1s = campoints1[j]

                Ti2 = robposes2[i]
                qi2 = np.roll(quaternion_from_matrix(Ti2), 1)
                ti2 = Ti2[:3, 3]
                pi2s = campoints2[i]

                Tj2 = robposes2[j]
                qj2 = np.roll(quaternion_from_matrix(Tj2), 1)
                tj2 = Tj2[:3, 3]
                pj2s = campoints2[j]

                for pi1, pj1, pi2, pj2 in zip(pi1s, pj1s, pi2s, pj2s):
                    solver.add_residual_block(qi1, ti1, pi1, qj1, tj1, pj1, qi2, ti2, pi2, qj2, tj2, pj2)

        q1opt, t1opt, q2opt, t2opt = solver.solve()
        X1opt = quaternion_matrix(np.roll(q1opt, -1))
        X2opt = quaternion_matrix(np.roll(q2opt, -1))
        X1opt[:3, 3] = t1opt
        X2opt[:3, 3] = t2opt

        summary = _eris.summary_to_dict(solver.summary())

        return X1opt, X2opt, summary

    def calibrate_eye_in_hand(self, problem, x1=None, x2 = None):
        return self._solve(problem, x1, x2)

    def calibrate_eye_to_hand(self, problem, x=None):
        problem.robposes = [inverse_matrix(robpose) for robpose in problem.robposes]
        return self._solve(problem, x)