import sys
from contextlib import closing
from six import StringIO
from gym import utils
import discrete
import numpy as np

# Map for wolf/prey scenario
MAP = [
    "+_   _   _+",
    "| | | | | |",
    "| |_| |_| |",
    "| : : : : |",
    "+---------+"
]


class WolfFixedGoalEnv(discrete.DiscreteEnv):
    """
    Wolf Simulation
    
    Curran Seam
    
    Description:
    There are 3 designated locations in the grid world indicated by Blue and Green. 
    When the episode starts, the Wolf starts off at a random square and the rabbit and hare are at 2 of the 3 designated locations. 
    The wolf navigates to the animal's locations then eats then animal. Once the animal is eaten, the episode ends.
    
    Observations: 
    There are 15 discrete states since there are 3 rows, 5 columns, 
    1 locations for the rabbit.
    
    Rabbit:
    Color: blue
    Reward: 1

    Actions:
    There are 4 discrete deterministic actions of the wolf:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    
    Rendering:
    - red: wolf
    - blue: rabbit
    
    state space is represented by:
        (wolf_row, wolf_col, rabbit_location)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')
        
        self.locs = locs = [(0, 4)]
        
        num_states = 15
        num_rows = 3
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 4 
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx1 in range(len(locs)):
                    state = self.encode(row, col, pass_idx1)
                    if pass_idx1 < 3:
                        initial_state_distrib[state] += 1

                    for action in range(num_actions):
                        # defaults
                        new_row, new_col, new_pass_idx1 = row, col, pass_idx1
                        reward = 0 # default reward
                        done = False

                        if action == 0:
                            new_row = min(row + 1, max_row)
                        elif action == 1:
                            new_row = max(row - 1, 0)
                        if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                            new_col = min(col + 1, max_col)
                        elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                            new_col = max(col - 1, 0)

                        new_state = self.encode(
                            new_row, new_col, new_pass_idx1)
                        if ((new_row, new_col) == locs[pass_idx1]):
                            done = True 
                            reward = 1
                        if state in [8, 6, 1, 3]:
                            P[state][action].append(
                                (0, new_state, reward, done))
                        else:
                            P[state][action].append(
                                (0.25, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, taxi_row, taxi_col, pass_loc1):
        # (3) 5, 3
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 1
        i += pass_loc1
        return i

    def decode(self, i):
        out = []
        out.append(i % 1)
        i = i // 1
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 3
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx1 = self.decode(self.s)

        def ul(x): return "_" if x == " " else x
        #if pass_idx1 < 3:
        out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
            out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
        p1i, p1j = self.locs[pass_idx1]
        out[1 + p1i][2 * p1j + 1] = utils.colorize(out[1 + p1i][2 * p1j + 1], 'cyan', highlight=True)
        '''
        p2i, p2j = self.locs[pass_idx2]
        out[1 + p2i][2 * p2j + 1] = utils.colorize(out[1 + p2i][2 * p2j + 1], 'green', highlight=True)
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)
        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        '''
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Eat"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
