import numpy as np
import pprint
import random

num_states = 11
num_actions = 11
q = np.zeros((num_states, num_actions)).tolist()

gamma = 0.8
random.seed()

'''
  9
__ __________
| 0 * 1 * 2 |
----------*--
| 3 * 4 * 5 |
--*-------*--
| 6 * 7 * 8 |
------ ------
      10
'''

r = [
    [-1, 0, -1, -1, -1, -1, -1, -1, 100, -1, -1],  # 0
    [0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],  # 1 ...
    [-1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1],
    [-1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1],
    [-1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, 0, -1, 0, -1, 100],
    [-1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1],
    [0, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1],
    [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, 100]
]

current_state = 4
prev_state = None

# print(r)

# while True:

path = []
maxx = 0
rewards = []


def next_state(states):
    global current_state
    l = []
    for i in range(len(states)):
        if states[i] > -1:
            l.append(i)
    return l


def make_step():
    global current_state, prev_state, r, q, maxx
    maxx = 0
    prob_next = 0
    possible_states = next_state(r[current_state])
    poss_st_copy = possible_states.copy()
    if prev_state is not None:
        possible_states.remove(prev_state)
    l = [r[current_state][possible_states[i]] for i in range(len(possible_states))]
    if l.count(l[0]) == len(l):
        i = random.randint(0, len(l) - 1)
        maxx = l[i]
        prob_next = possible_states[i]
    else:
        for i in range(len(l)):
            if l[i] >= maxx:
                maxx = l[i]
                prob_next = possible_states[i]

    if prev_state is not None:
        q[prev_state][current_state] = r[prev_state][current_state] + \
                                       gamma * max([q[current_state][j] for j in poss_st_copy])
    if prev_state == current_state:
        q[prev_state][current_state] = r[prev_state][current_state] + \
                                       gamma * max([q[current_state][j] for j in poss_st_copy])
        current_state = 4
        prev_state = None
        return
    prev_state = current_state
    current_state = prob_next

    rewards.append(q[prev_state][current_state])
    path.append((prev_state, current_state))


for i in range(10):
    make_step()
print(path)
pprint.pprint(q)
