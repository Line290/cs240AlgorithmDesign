import numpy as np
import numpy as np
import copy
def gradebot(answer):
	return int(1e9 + 1e6 * np.log(answer))

x = 5.19644700049e-27
print gradebot(x)
y = 939500278
an = np.exp((y - 1e9)*1.0/1e6)
print an

f = open("123","r")
lines = f.readlines()
x =[]
i = 0
for j ,line in enumerate(lines):
	for k, num in enumerate(line.split()):
		x.append(int(num))
# print x
# print ' '.join('%+d' % i for i in tuple(x))

def power_by_mtt(state, edges):
    """Calculate the total power of the state, by the matrix-tree theorem.
    """
    # import numpy as np

    n = len(state)
    graph = np.zeros((n+1, n+1), dtype=np.float64)
    for u, v, w in edges:
        if (u == 0 or u in state) and v in state:
            graph[abs(u), abs(v)] = w
    mat_l = np.zeros((n+1, n+1), dtype=np.float64)
    for i in range(n+1):
        for j in range(n+1):
            if i == j:
                for k in range(n+1):
                    if k != i:
                        mat_l[i, j] += graph[k, i]
            else:
                mat_l[i, j] = -graph[i, j]
    det = np.linalg.det(mat_l[1:, 1:])
    return det
def two_step_search(state, power):
    n, edges = read_input()

    best_state = copy.copy(state)
    best_power = power
    for i in range(n):
        print i
        for j in range(i+1, n):
            temp_state = copy.copy(best_state)
            temp_state[i] *= -1
            temp_state[j] *= -1
            temp_power = power_by_mtt(tuple(temp_state), edges)
            if temp_power > best_power:
                best_state = temp_state
                best_power = temp_power
                print best_power
    print ' '.join('%+d' % i for i in tuple(best_state))
def three_step_search(state, power):
    n, edges = read_input()

    best_state = copy.copy(state)
    best_power = power
    for i in range(n-2):
        # print "i:" + i
        for k in range(i+1, n-1):
            print "i: " + str(i) + " j: " + str(k)
            for j in range(k+2, n):
                temp_state = copy.copy(best_state)
                temp_state[i] *= -1
                temp_state[j] *= -1
                temp_state[k] *= -1
                temp_power = power_by_mtt(tuple(temp_state), edges)
                if temp_power > best_power:
                    best_state = temp_state
                    best_power = temp_power
                    print best_power
    print ' '.join('%+d' % i for i in tuple(best_state))
# def greed_algorithm():
#     # import numpy as np
#     # import copy

#     n, edges = read_input()
#     # greed search
#     temp_state = []
#     for i in range(1,n+1):
#         print i
#         temp_state.append(i)
#         temp_state_p = copy.copy(temp_state)
#         temp_state.pop()
#         temp_state.append(-i)
#         temp_state_n = copy.copy(temp_state)
#         temp_state.pop()
#         temp_power_p = power_by_mtt(tuple(temp_state_p), edges)
#         temp_power_n = power_by_mtt(tuple(temp_state_n), edges)
#         if temp_power_p > temp_power_n:
#             temp_state.append(i)
#         else:
#             temp_state.append(-i)
#     temp_power = power_by_mtt(tuple(temp_state), edges)
#     return temp_state, temp_power


# # def back_greed(state, power):
#     # n, edges = read_input()
#  #    # greed search
#  #    temp_state = []
#  #    weight = np.zeros((2*n+1, 2*n+1), np.float64)
# 	# for u, v, w in edges:
#  #    	weight[u+n, v+n] = w
# 	# i = 0
# 	# visited = np.zeros(n+1, np.int32)
# 	# dis = np.zeros(n+1, np.float64)
# 	# adj = np.zeros(n+1, np.int32)
# 	# for i in range(n+1):
# 	# 	adj[i] = 0
# 	# 	if 
# def edge_greed():

# 	n, edges = read_input()

# 	sort_edges = sorted(edges,cmp=lambda x,y : cmp(x[2], y[2]))
# 	sort_edges.reverse()
# 	visited = np.zeros(n+1, np.int32)
# 	sign = np.zeros(n+1, np.int32)
# 	for u, v, w in sort_edges:
# 	    if visited[abs(u)] == 0 and visited[abs(v)] == 0:
# 	        visited[abs(u)] = 1
# 	        visited[abs(v)] = 1
# 	        if u < 0:
# 	            sign[abs(u)] = -1
# 	        else:
# 	            sign[abs(u)] = 1
# 	        if v < 0:
# 	            sign[abs(v)] = -1
# 	        else:
# 	            sign[abs(v)] = 1
# 		if visited.sum() == n+1 and (n+1)%2 == 0:
# 			print 'ff'
# 			break
# 		elif visited.sum() == n and (n+1)%2 == 1:
# 			alone = np.where(visited == 0)
# 			alone = alone[0]
# 			for u, v, w in sort_edges:
# 				if abs(u) == alone and v * sign[v] > 0:
# 					sign[abs(u)] = u/abs(u)
# 				elif u * sign[u] > 0 and abs(v) == alone:
# 					sign[abs(v)] = v/abs(v)

# 		# n+1 is odd

# 	state = np.array(range(1,n+1)) * sign[1:]
# 	state_tuple =  tuple(state)
# 	power = power_by_mtt(state_tuple, edges)
# 	print power
# 	return list(state), power

def local_opt_search(init_state, init_power):
    n, edges = read_input()

    state = copy.copy(init_state)
    power = init_power
    flag = False
    while flag == False:
        flag = True
        for i in range(n):
            print i
            state_update = copy.copy(state)
            state_update[i] *= -1
            state_update_tuple = tuple(state_update)
            power_update = power_by_mtt(state_update_tuple, edges)
            if power_update > power:
                state = copy.copy(state_update)
                power = power_update
                flag = False
                print power
    print ' '.join('%+d' % i for i in tuple(state))
    # return state, power

def read_input():
    f = open("2","r")
    lines = f.readlines()
    edges = []
    for i, line in enumerate(lines):
        if i == 0:
            n = int(line)
            continue
        u, v, w = line.split()
        edges.append((int(u), int(v), float(w)))
    return n, edges
if __name__ == '__main__':
    n, edges = read_input()
    power = power_by_mtt(x, edges)
    print power
    # two_step_search(x, power)
    three_step_search(x, power)
# if __name__ == '__main__':
#     # randomized_algorithm()
#     # init_state, init_power = greed_algorithm()
#     # print 'greed:'
#     # print init_power
#     # print ' '.join('%+d' % i for i in tuple(init_state))
#     init_state, init_power = edge_greed()
#     print 'greed:'
#     print init_power
#     print ' '.join('%+d' % i for i in tuple(init_state))
#     state, power = local_opt_search(init_state, init_power)
#     print 'greed + local:'
#     print power
#     print ' '.join('%+d' % i for i in tuple(state))
# # def power_by_mtt(state, edges):
# #     """Calculate the total power of the state, by the matrix-tree theorem.
# #     """
# #     # import numpy as np

# #     n = len(state)
# #     graph = np.zeros((n+1, n+1), dtype=np.float64)
# #     for u, v, w in edges:
# #         if (u == 0 or u in state) and v in state:
# #             graph[abs(u), abs(v)] = w
# #     mat_l = np.zeros((n+1, n+1), dtype=np.float64)
# #     for i in range(n+1):
# #         for j in range(n+1):
# #             if i == j:
# #                 for k in range(n+1):
# #                     if k != i:
# #                         mat_l[i, j] += graph[k, i]
# #             else:
# #                 mat_l[i, j] = -graph[i, j]
# #     det = np.linalg.det(mat_l[1:, 1:])
# #     return det
# # n, edges = read_input()
# # power =  power_by_mtt(tuple(x), edges)
# # print power
# # print gradebot(power)