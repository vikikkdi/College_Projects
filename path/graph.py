from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import numpy as np
from hungarian import Hungarian
from time import sleep
import coordinates

import numpy as np
import math
import pppp


def pseudo_cost_function(initial_pos, desired_shape, n):
	k = [[0 for j in range(n)] for i in range(n)]

	for i in range(n):
		for j in range(n):
			x = -np.transpose(initial_pos[i]).dot(np.array(desired_shape[j]))
			k[i][j] = x

	hungarian = Hungarian(k)
	hungarian.calculate()
	x_star = hungarian.get_results()
	k_star = hungarian.get_total_potential()

	return [x_star, k_star]

def optimal_goal_formation(initial_pos, desired_shape, x_star):
	d = pppp.gradient_descent(initial_pos, desired_shape, x_star)
	#qq = np.array(d) + np.array(desired_shape)
	return d

n=10
initial_pos = coordinates.generate_initial(n)
desired_shape = coordinates.generate_diamond(n)

a = pseudo_cost_function(initial_pos, desired_shape, n)
x_star = a[0]
k_star = a[1]

qq = optimal_goal_formation(initial_pos, desired_shape, x_star)


fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(qq[0], qq[1], qq[2])

ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Value of cost function (meters)')
pyplot.show()