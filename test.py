import numpy as np
import random
import copy
def gradebot(answer):
	return int(1e9 + 1e6 * np.log(answer))

y = 909384264
an = np.exp((y - 1e9)*1.0/1e6)
print an

# f = open("1234","r")
# lines = f.readlines()
# x =[]
# i = 0
# for j ,line in enumerate(lines):
# 	for k, num in enumerate(line.split()):
# 		x.append(int(num))
# print x
# print ' '.join('%+d' % i for i in tuple(x))

def read_input():
    f = open("4","r")
    lines = f.readlines()
    edges = []
    for i, line in enumerate(lines):
        if i == 0:
            n = int(line)
            continue
        u, v, w = line.split()
        edges.append((int(u), int(v), float(w)))
    return n, edges

def mp():
    '''Create a edge matrix which shapes is (2n+1, 2n+1)
    '''
    n, edges = read_input()
    edges_mp = np.zeros((2*n+1, 2*n+1), np.float64)
    for u, v, w in edges:
        edges_mp[u+n, v+n] = w
    return edges_mp, n

global edges_mp, N
edges_mp, N = mp()

def power_by_mtt(state):
    """Calculate the total power of the state, by the matrix-tree theorem.
    """
    state = np.asarray(state)
    n = state.shape[0]
    graph = np.zeros((n+1, n+1), dtype=np.float64)
    index = np.zeros(n+1, np.int32)
    # Include node 0
    index[1:] = state
    # Offset
    index += N
    graph = edges_mp[index.reshape(n+1, -1), index]

    mat_l = np.diag(np.sum(graph, axis = 0))
    mat_l -= graph

    det = np.linalg.det(mat_l[1:, 1:])
    return det

def SA(init_state, init_power):
    random.seed(0)

    state = copy.copy(init_state)
    power = init_power
    flag = False
    while flag == False:
        # flag = True
        for i in range(N):
            print i
            state_update = copy.copy(state)
            state_update[i] *= -1
            state_update_tuple = tuple(state_update)
            power_update = power_by_mtt(state_update_tuple)
            if power_update > power or random.random() < 0.01:
            	if power_update > 1.0e-27:
                    state = copy.copy(state_update)
                    power = power_update
                if power_update > 5.2e-27:
                    flag = True
                print power
    print "one step"
    print ' '.join('%+d' % i for i in tuple(state))
    return state, power

def two_step_search(state, power):

    best_state = copy.copy(state)
    best_power = power
    for i in range(N):
        print i
        for j in range(i+1, N):
            temp_state = copy.copy(best_state)
            temp_state[i] *= -1
            temp_state[j] *= -1
            temp_power = power_by_mtt(tuple(temp_state))
            if temp_power > best_power:
                best_state = temp_state
                best_power = temp_power
                print best_power
    print "two step"
    print ' '.join('%+d' % i for i in tuple(best_state))
    return best_state, best_power

def three_step_search(state, power):
    n, edges = read_input()

    best_state = copy.copy(state)
    best_power = power
    for i in range(44, N-2):

        for k in range(i+1, N-1):
            
            for j in range(k+1, N):
            	print "i: " + str(i) + " j: " + str(k) + " k: " + str(j)
                temp_state = copy.copy(best_state)
                temp_state[i] *= -1
                temp_state[j] *= -1
                temp_state[k] *= -1
                temp_power = power_by_mtt(tuple(temp_state))
                print temp_power
                if temp_power > best_power:
                    best_state = temp_state
                    best_power = temp_power
                    print best_power
    print ' '.join('%+d' % i for i in tuple(best_state))
def greed_algorithm():

    # greed search
    temp_state = []
    for i in range(1,N+1):
        print i
        temp_state.append(i)
        temp_state_p = copy.copy(temp_state)
        temp_state.pop()
        temp_state.append(-i)
        temp_state_n = copy.copy(temp_state)
        temp_state.pop()
        temp_power_p = power_by_mtt(tuple(temp_state_p))
        temp_power_n = power_by_mtt(tuple(temp_state_n))
        if temp_power_p > temp_power_n:
            temp_state.append(i)
        else:
            temp_state.append(-i)
    temp_power = power_by_mtt(tuple(temp_state))
    return temp_state, temp_power

# def edge_greed():

#   n, edges = read_input()

# 	sort_edges = sorted(edges,cmp=lambda x,y : cmp(x[2], y[2]))
# 	sort_edges.reverse()
# 	visited = np.zeros(N+1, np.int32)
# 	sign = np.zeros(N+1, np.int32)
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
# 		if visited.sum() == N+1 and (N+1)%2 == 0:
# 			print 'ff'
# 			break
# 		elif visited.sum() == N and (N+1)%2 == 1:
# 			alone = np.where(visited == 0)
# 			alone = alone[0]
# 			for u, v, w in sort_edges:
# 				if abs(u) == alone and v * sign[v] > 0:
# 					sign[abs(u)] = u/abs(u)
# 				elif u * sign[u] > 0 and abs(v) == alone:
# 					sign[abs(v)] = v/abs(v)

# 		# N+1 is odd

# 	state = np.array(range(1,N+1)) * sign[1:]
# 	state_tuple =  tuple(state)
# 	power = power_by_mtt(state_tuple)
# 	print power
# 	return list(state), power
def two_step_max_search(state, power):

    best_state = copy.copy(state)
    best_power = power
    for i in range(N):
        print i
        for j in range(i+1, N):
            temp_state = copy.copy(state)
            temp_state[i] *= -1
            temp_state[j] *= -1
            temp_power = power_by_mtt(tuple(temp_state))
            if temp_power > best_power:
                best_state = temp_state
                best_power = temp_power
                print best_power
    print "two step"
    print ' '.join('%+d' % i for i in tuple(best_state))
    return best_state, best_power
def local_max_search(init_state, init_power):
    n, edges = read_input()

    state = copy.copy(init_state)
    power = init_power
    flag = False
    while flag == False:
        flag = True
        for i in range(N):
            print i
            state_update = copy.copy(init_state)
            state_update[i] *= -1
            state_update_tuple = tuple(state_update)
            power_update = power_by_mtt(state_update_tuple)
            if power_update > power:
                state = copy.copy(state_update)
                power = power_update
                flag = False
                print power
        init_state = copy.copy(state)
        init_power = power
        print "one step max power"
        print power
    print "one step"
    print ' '.join('%+d' % i for i in tuple(state))
    return state, power
def local_opt_search(init_state, init_power):
    n, edges = read_input()

    state = copy.copy(init_state)
    power = init_power
    flag = False
    while flag == False:
        flag = True
        for i in range(N):
            print i
            state_update = copy.copy(state)
            state_update[i] *= -1
            state_update_tuple = tuple(state_update)
            power_update = power_by_mtt(state_update_tuple)
            if power_update > power:
                state = copy.copy(state_update)
                power = power_update
                flag = False
                print power
                # print state
    print "one step"
    print ' '.join('%+d' % i for i in tuple(state))
    return state, power
# def one_step_down(local_opt_state, local_opt_power):
#     n, edges = read_input()
#     all_one_step_state = None
#     for i in range(n):
#         temp_state = copy.copy(local_opt_state)
#         temp_state[i] *= -1
#         temp_power = power_by_mtt(tuple(temp_state), edges)
#         all_one_step_state.append((temp_state, temp_power))
#     return all_one_step_state, local_opt_state, local_opt_power
# def climb_again(all_one_step_state, local_opt_state, local_opt_power):
#     n, edges = read_input()
#     for i in range(n):
#         temp_state, temp_power = all_one_step_state[i]
#         for j in range(n):


if __name__ == '__main__':
    # n, edges = read_input()
    # power = power_by_mtt(tuple(x), edges)
    # print power
    state, power = greed_algorithm()
    print "greed:"
    print power
    print state
    state, power = local_opt_search(state, power)
    print state
    print power
    # two_step_max_search(x, power)
    # SA(x, power)
    # state, power = greed_algorithm()
    # print state
    # print power
    # one_step_state, one_step_power = local_max_search(state, power)
    # two_step_state, two_step_power = two_step_search(one_step_state, one_step_power)
    # one_step_state, one_step_power = local_max_search(two_step_state, two_step_power)
    # power = power_by_mtt(x, edges)
    # print power
    # # two_step_search(x, power)
    # three_step_search(x, power)
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