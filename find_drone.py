import matplotlib.pyplot as plt
import numpy as np
import copy
import heapq
from queue import deque

'''
To Do : 

Check and change performAction
Change logic in Astar
Change heuristics to get different length sequence
Change manhattan distance to bfs

change code for bfs
'''


class Finder:
    def __init__(self):
        self.distanceMatrix = None
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

    def convertToTuple(self, grid):
        lst = []

        for i in grid:
            lst.append(tuple(i))

        return hash(tuple(lst))

    def bfs(self, destination, grid):

        queue = deque()

        queue.append((destination, 0))

        visited = set()
        visited.add(destination)

        rows = [-1, 1, 0, 0]
        cols = [0, 0, -1, 1]
        path = [[-1 for i in range(len(grid[0]))] for j in range(len(grid))]
        directions = ["DOWN", "UP", "RIGHT", "LEFT"]

        while queue:
            elem = queue.popleft()
            row = elem[0][0]
            col = elem[0][1]
            for i in range(4):
                newRow = row + rows[i]
                newCol = col + cols[i]

                if (
                    0 <= newRow < len(grid)
                    and 0 <= newCol < len(grid[0])
                    and (newRow, newCol) not in visited
                    and grid[newRow][newCol] != -1
                ):
                    visited.add((newRow, newCol))
                    path[newRow][newCol] = elem[1] + 1
                    queue.append(((newRow, newCol), elem[1] + 1))

        return path

    def fourMatrix(self, grid):
        movi = [
            [
                [[0 for i in range(len(grid[0]))] for j in range(len(grid))]
                for k in range(len(grid[0]))
            ]
            for l in range(len(grid))
        ]

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                path = self.bfs((i, j), grid)
                for k in range(len(path)):
                    for l in range(len(path[0])):
                        movi[i][j][k][l] = path[k][l]

        self.distanceMatrix = movi

    def compute_heuristic(self, probs, actions):
        heuristic = 0

        #one of the heuristic I am using is the max distance between any two nodes with non-zero probability.
        # I computed this by iterating through the entire array and checking for distance.
        maxdist = 0
        for i in range(len(probs)):
            for j in range(len(probs[0])):
                d = 0
                #compute only if it has probability more than 0
                if probs[i][j] > 0:
                    for p in range(len(probs)):
                        for q in range(len(probs[0])):
                            if probs[p][q] > 0:
                                
                                #distance between these two nodes
                                if not self.distanceMatrix:
                                    
                                    self.fourMatrix(self.probs) 
                                
                                manhattan = self.distanceMatrix[p][q][i][j]
                                # calculate manhattan distance
                                # manhattan = abs(p - i) + abs(q - j)

                                d = max(d, manhattan)

                maxdist = max(maxdist, d)

        ctr = 0
        for i in probs:
            for j in i:
                if j > 0:
                    ctr += 1
        
        #add weights to heuristic
        return ( 80 * (maxdist ) ) * ((10 * ctr)) * ( len(actions))

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

    def testActions( self, probs, actions):
        print("before: ")
        self.display(probs)
        newprobs = copy.deepcopy(probs)
        print(actions)
        for action in actions:
            
            if action == "UP":
                newGrid = self.move_up(newprobs)
            elif action == "DOWN":
                newGrid = self.move_down(newprobs)
            elif action == "LEFT":
                newGrid = self.move_left(newprobs)
            elif action == "RIGHT":
                newGrid = self.move_right(newprobs)
            else:
                newGrid = copy.deepcopy(newprobs)

            newprobs = copy.deepcopy(newGrid)
        print("after: ")
        self.display(newGrid)
        
        return newGrid

    def astar(self):

        heap = []
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        probs = copy.deepcopy(self.probs)

        h = self.compute_heuristic(probs,[])

        heap.append((h, [], probs))

        visited = set()

        visited.add(hash(self.convertToTuple(probs)))
        while heap:
            elem = heapq.heappop(heap)

            if self.check(elem[2]):
                # print(elem[2])
                return elem[1]

            actions = elem[1]
            probs = elem[2]
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

                newprobs = self.performAction(probs, act)

                newHeu = self.compute_heuristic(newprobs,actions)

                if hash(self.convertToTuple(newprobs)) not in visited:
                    visited.add(hash(self.convertToTuple(newprobs)))
                    heapq.heappush(heap, (newHeu, actions[:] + [act], newprobs))

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
        with open("schematic.txt", "r") as file:
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

    # print(f.probs)
    
    # newprobs = f.move_down(f.probs)
    # print(newprobs)
    print(f.find_drone())
    # print(f.probs)
    # actions = ['LEFT', 'LEFT', 'UP', 'UP', 'UP', 'UP', 'LEFT', 'LEFT', 'LEFT', 'UP', 'UP', 'LEFT', 'LEFT', 'UP', 'UP', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'RIGHT', 'UP', 'LEFT', 'UP', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'UP', 'LEFT', 'DOWN', 'DOWN', 'RIGHT', 'DOWN', 'DOWN', 'DOWN', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'UP', 'UP', 'UP', 'RIGHT', 'UP', 'LEFT', 'UP', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'RIGHT', 'DOWN', 'DOWN', 'LEFT', 'DOWN', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'UP', 'RIGHT', 'RIGHT', 'UP', 'UP', 'UP', 'UP', 'UP', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'UP', 'RIGHT', 'DOWN', 'DOWN', 'LEFT', 'UP', 'LEFT', 'LEFT', 'DOWN', 'RIGHT', 'RIGHT', 'UP', 'LEFT', 'UP', 'UP', 'UP', 'UP', 'UP', 'LEFT', 'UP', 'RIGHT', 'UP', 'RIGHT', 'UP', 'LEFT', 'DOWN', 'LEFT', 'UP', 'LEFT', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP', 'UP']
    # f.testActions(f.probs,actions)