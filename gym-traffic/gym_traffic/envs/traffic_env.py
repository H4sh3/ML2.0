import itertools
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym.envs.toy_text import discrete
from io import StringIO
import sys
from contextlib import closing
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

def gen_permutation(positions):
    res = []
    for i in range(len(positions)+1):
        for x in itertools.combinations(positions,i):
            tmp = []
            for o in x:
                tmp.append(o)
            res.append(tmp)
    return res

class TrafficEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        num_rows = 10
        num_cols = 10
        self.shape = (num_rows, num_cols)
        self.nA = 4
        self.iter_count = 0

        self.reset()
        self.nS = num_rows*num_cols*len(self.coin_positions)
        '''P = {state: {action: []
                for action in range(self.nA)} for state in range(self.nS)}
'''
        P = {}
        for row in range(num_rows):
            for col in range(num_cols):
                for coin_pos in gen_permutation(self.coin_positions):
                    state = self.encode_state(row, col, coin_pos)
                    if [row,col] in coin_pos:
                        coin_pos.remove([row,col])
                    P[str(state)] = { a: [] for a in range(self.nA)}
                    P[str(state)][UP] = self.calc_trans_prob(row, col, [0,-1],coin_pos)
                    P[str(state)][DOWN] = self.calc_trans_prob(row, col, [0,1],coin_pos)
                    P[str(state)][LEFT] = self.calc_trans_prob(row, col, [-1,0],coin_pos)
                    P[str(state)][RIGHT] = self.calc_trans_prob(row, col, [1,0],coin_pos)
        self.states = P.keys()
        self.reset()
        isd = self.encode_state(0, 0, self.coin_positions)
        super(TrafficEnv, self).__init__(self.nS, self.nA, P, isd)

    
    def encode_state(self, row, col, coin_pos):
        pos_suffix = ''
        if len(coin_pos) > 0:
            for p in coin_pos:
                if len(p) > 0:
                    pos_suffix+=str(p[0])
                    pos_suffix+=str(p[1])
                else:
                    pos_suffix = '0'
        else:
            pos_suffix = '0'
            
        return 'x'+str(col)+'y'+str(row)+'pos'+pos_suffix

    def reset(self):
        self.iter_count = 0
        self.collected_coins = 0
        self.coin_positions = [[2,2],[8,8],[1,9],[9,0]]

    def get_initial_state(self):
        return self.encode_state(0,0,self.coin_positions)


    def calc_trans_prob(self, row, col, delta,coin_positions):
        new_pos = np.array((row, col)) + np.array(delta)
        before = new_pos.copy()
        new_pos = self._limit_coordinates(new_pos).astype(int)
        reward = -10
        if (before[0] != new_pos[0] or before[1] != new_pos[1]):
            # got changed by limit coords function
            reward = -10

        new_state = self.encode_state(new_pos[0],new_pos[1],coin_positions)
        
        if(len(coin_positions) == 4):
            if (new_pos[0] == coin_positions[0][0] and new_pos[1] == coin_positions[0][1]):
                reward = 150 
        else:
            reward = 100 if [new_pos[0],new_pos[1]] in coin_positions else reward

        # done if no more coins positions in encoded state
        done = new_state.split("pos",1)[1] == '0'
        return [(1.0, new_state, reward, done)]

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def render(self, state, mode="human"):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        output = ''
        for y in range(self.shape[0]):
            output = output.rstrip()
            output += '\n'
            for x in range(self.shape[1]):
                position = [x,y]
                p_x = state.split("x",1)[1].split("y",1)[0]
                p_y = state.split("y",1)[1].split("p",1)[0]
                if str(x) == p_x and str(y) == p_y:
                    output = " X "
                elif position in self.coin_positions:
                    output = " C "
                else:
                    output = " o "

                outfile.write(output)
            outfile.write('\n')
        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()