import numpy as np
np.set_printoptions(suppress=True)
from glob import glob

import eris
from eris.transformations import quaternion_matrix, inverse_matrix
from eris.problem2 import Problem2
from eris.solver2 import Solver2

def random_pose():
    q = np.random.rand(4)
    q /= np.linalg.norm(q)
    T = quaternion_matrix(q)
    T[:3, 3] = np.random.rand(3)
    return T


def chessboard_corners(pattern_size=(10, 6), square_size=0.02):
    pattern_points = np.zeros((np.prod(pattern_size), 4), np.float64)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points[:, :2] *= square_size
    pattern_points[:, -1] += 1.0
    return pattern_points.T


P = chessboard_corners()  # Points in chessboard
T = random_pose()  # Chessboard in base
X = random_pose()  # Camera in end-effector
Xinv = inverse_matrix(X)

num_samples = 10
robposes = [random_pose() for _ in range(num_samples)]

campoints = [(Xinv @ inverse_matrix(A) @ T @ P)[:3, :].T for A in robposes]
    
campoints_true = []
for point in P.T:
    true_point = T  @ point.T
    campoints_true.append(true_point[:-1])

campts_true = []
for i in range(len(campoints)):
    campts_true.append(campoints_true)

problem = eris.Problem2(campoints = campoints, campoints_true = campts_true, robposes =robposes)
solver = eris.Solver2()

sol, summary = solver.calibrate_eye_in_hand(problem)

print(summary["full_report"])
print(X)
print(sol)
print(X @ inverse_matrix(sol))

