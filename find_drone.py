import matplotlib.pyplot as plt
import numpy as np
import copy
import heapq


class Finder:
    def __init__(self):
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        return

    def check(self, grid):

        cnt = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] > 0:
                    cnt += 1
        return False if cnt > 1 else True

    def find_drone(self):

        ans = self.astar()
        print(len(ans))
        return ans

    def aStarSearch(self, grid):
        fringe = []
        heapq.heapify(fringe)
        self.sequence = []
        self.actions = ['R', 'D', 'L', 'U']
        self.visited = set()

        '''
        add initial state to fringe

        loop start

        

        compute possible next states based on actions.
        calculate heuristic for these states and add into fringe.

        Take from priority queue, the state with least value and move to that state.
        Add the action to sequence.

        check if this state is goal state. if true return sequence.

        Store probability array also in heap
        '''

    

        return
    '''
    ================================CHANGE BELOW CODE===========================
    '''

    def convertToTuple(self, grid):
        lst = []

        for i in grid:
            lst.append(tuple(i))

        return hash(tuple(lst))

    def heuristic(self, grid, actions):

        return (50 * self.computeMaxDistance(grid)) * (5 * self.printNonZeroCount(grid))
    
    def computeMaxDistance(self, grid):

        maxDist = 0
        n = len(grid)
        m = len(grid[0])

        n = len(grid)
        for i in range(n):
            for j in range(m):
                d = 0
                if grid[i][j] > 0:
                    for k in range(n):
                        for l in range(m):
                            if grid[k][l] > 0:
                                d = max(d, self.computeDistance((i, j), (k, l)))

                maxDist = max(maxDist, d)

        return maxDist

    def computeDistance(self, p1, p2):
        return abs(p2[1] - p1[1]) + abs(p2[0] - p1[0])
    
    def printNonZeroCount(self, grid):
        ctr = 0
        for i in grid:
            for j in i:
                if j > 0:
                    ctr += 1

        return ctr

    def performAction(self, grid, action=[]):
        if type(action) == list:
            newGrid = copy.deepcopy(grid)
            for act in action:
                newGrid = self.performAction(newGrid, act)

            return newGrid
        else:
            if action == "UP":
                newGrid = self.move_up(grid)
            elif action == "DOWN":
                newGrid = self.move_down(grid)
            elif action == "LEFT":
                newGrid = self.move_left(grid)
            elif action == "RIGHT":
                newGrid = self.move_right(grid)
            else:
                newGrid = copy.deepcopy(grid)

            return newGrid

    def astar(self):

        heap = []

        grid = copy.deepcopy(self.probs)

        h = self.heuristic(grid, [])

        heap.append((h, [], grid))

        visited = set()

        visited.add(hash(self.convertToTuple(grid)))
        while heap:
            elem = heapq.heappop(heap)
            # print(elem)

            if self.check(elem[2]):
                return elem[1]

            actions = elem[1]
            grid = elem[2]
            heu = elem[0]

            for act in self.actions:
                if actions and actions[-1] == "LEFT" and act == "RIGHT":
                    continue
                if actions and actions[-1] == "RIGHT" and act == "LEFT":
                    continue

                if actions and actions[-1] == "UP" and act == "DOWN":
                    continue

                if actions and actions[-1] == "DOWN" and act == "UP":
                    continue

                newGrid = self.performAction(grid, act)

                newHeu = self.heuristic(newGrid, actions + [act])

                if hash(self.convertToTuple(newGrid)) not in visited:
                    visited.add(hash(self.convertToTuple(newGrid)))
                    heapq.heappush(heap, (newHeu, actions[:] + [act], newGrid))

        return []

    def move_right(self, probs):
        newprobs = copy.deepcopy(probs)
        for i in range(len(self.probs)):
            for j in range(len(self.probs[0])-1,0,-1):
                if  newprobs[i][j] != -1: 
                    if newprobs[i][j-1] != -1:
                        newprobs[i][j] += newprobs[i][j-1]
                        newprobs[i][j-1] = 0
        return newprobs

    def move_left(self, probs):
        
        newprobs = copy.deepcopy(probs)
        for i in range(len(self.probs)):
            for j in range(len(self.probs[0])-1):
                if newprobs[i][j] != -1:
                    if newprobs[i][j+1] != -1:
                        newprobs[i][j] += newprobs[i][j+1]
                        newprobs[i][j+1] = 0

        return newprobs
    
    def move_up(self, probs):

        newprobs = copy.deepcopy(probs)

        for i in range(len(self.probs)-1):
            for j in range(len(self.probs[0])):

                if newprobs[i][j] != -1:
                    if newprobs[i+1][j] != -1:

                        newprobs[i][j] += newprobs[i+1][j]
                        newprobs[i+1][j] = 0
        return newprobs
    
    def move_down(self, probs):

        newprobs = copy.deepcopy(probs)

        for i in range(len(self.probs)-1,0,-1):
            for j in range(len(self.probs[0])):

                if newprobs[i][j] != -1:
                    if newprobs[i-1][j] != -1:
                        newprobs[i][j] += newprobs[i-1][j]
                        newprobs[i-1][j] = 0
        return newprobs

    def load_grid(self):
        
        self.grid = []
        self.opencells = 0
        with open("input2.txt", "r") as file:
            reader = file.readlines()

            for r in reader:
                # print(r)
                # print("line len: ", len(r))
                self.temp = []
                for cell in r:
                    if cell == '_':
                        self.temp.append(0)
                        self.opencells += 1
                    elif cell == "X":
                        self.temp.append(1)
                
                self.grid.append(self.temp)

        self.probs = []
        for i in range (len(self.grid)):
            self.temp = []
            for j in range (len(self.grid[i])):
                if self.grid[i][j] == 0:
                    dummy = 1/self.opencells
                    self.temp.append(dummy)
                else:
                    self.temp.append(-1)
            self.probs.append(self.temp)

    def display(self, probs):
        for row in probs:
            print("\t".join(map(str, row)))
            
if __name__ == "__main__":

    f = Finder()

    f.load_grid()

    print(f.find_drone())
