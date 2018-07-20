import random
import pandas as pd
import numpy as np

GAMMA = 0.2
EPSILONE = 0.1
EPISODES = 1E4
MAPFILE = "state.csv"
QFILE = "q-table.csv"

lines = open(MAPFILE, "r").readlines()
r = []
for line in lines:
    line = [int(i.strip()) for i in line.split(",")]
    r.append(line)
r = np.array(r).astype("float32")
q = np.zeros_like(r)


def write_q():
    q_file = open(QFILE, "w")
    tmp = ", "
    for i in range(q.shape[0] - 1):
        tmp += str(i) + ", "
    tmp += str(q.shape[0] - 1)
    q_file.write(tmp + "\n")

    for i in range(q.shape[0]):
        line = "%d, " % i
        for j in range(q[i].shape[0] - 1):
            line += "%.3f, " % round(float(q[i][j]), 3)
        line += str(
            round(
                float(q[i][q[i].shape[0] - 1]),
                3
            )
        )
        q_file.write(line + "\n")


def get_action(current_state, possible_actions,):
    p = random.random()
    if p >= EPSILONE:
        estimated_rewards = []
        for possible_action in possible_actions:
            estimated_rewards.append(q[current_state][possible_action])
        ers = np.argwhere(estimated_rewards == np.amax(estimated_rewards))
        er = random.choice(ers)
        action = (current_state, possible_actions[er[0]])

    else:
        action = (current_state, random.choice(possible_actions))

    return action


def get_posible_actions(current_state):
    possible_actions = []
    for index, valid in enumerate(r[current_state] >= -1):
        if valid:
            possible_actions.append(index)
    return possible_actions

def get_best_action(self, state):
 
        # Return the action (index) with maximum Q-value
        return self._qmatrix[[state]].idxmax().iloc[0]

# def update_model(self, state, action, reward, next_state):

#     # Update q_value for a state-action pair Q(s,a):
#     # Q(s,a) = Q(s,a) + α( r + γmaxa' Q(s',a') - Q(s,a) )
#     q_sa = self._qmatrix.ix[action, state]
#     max_q_sa_next = self._qmatrix.ix[self.get_best_action(next_state), next_state]
#     r = reward
#     alpha = self._learn_rate
#     gamma = self._discount_factor

#     # Do the computation
#     new_q_sa = q_sa + alpha * (r + gamma * max_q_sa_next - q_sa)
#     self._qmatrix.set_value(action, state, new_q_sa)

valid_states = []
banned_states = []
goal_state = None
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if r[i][j] == -1:
            banned_states.append(j)
        elif r[i][j] >= -1:
            valid_states.append(j)
        if r[i][j] == 10:
            goal_state = j

valid_states = list(set(valid_states))
banned_states = list(set(banned_states))

for episode in range(int(EPISODES)):
    initial_state = random.choice(valid_states)
    current_state = initial_state
    print("Started from initial state: %d" % current_state)
    chain = []
    goal = True if current_state == goal_state else False

    while not goal:
        chain.append(current_state)
        possible_actions = get_posible_actions(current_state)
        action = get_action(
            current_state,
            possible_actions
        )
        next_state = action[1]

        max_q = np.argmax(q[next_state])
        q[current_state][next_state] += ALPHA * (
            r[current_state][next_state] + GAMMA *
            q[next_state][action_prime] - q[current_state][next_state]

        if next_state not in banned_states:
            current_state = next_state
            goal = True if current_state == goal_state else False
    if current_state != goal_state:
        if chain[-1] != current_state:
            chain.append(current_state)
    else:
        chain.append(current_state)
    print(' --> '.join(str(x) for x in chain))


write_q()



