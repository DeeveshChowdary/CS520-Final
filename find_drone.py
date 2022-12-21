import matplotlib.pyplot as plt
import numpy as np
import copy
import heapq


class Finder:
    def __init__(self):
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

    def getNeighbors(self,cell,probs):
        neighbors = []
        row = cell[0]
        col = cell[1]

        #check top neighbor
        if row-1 >= 0:
            if probs[row-1][col] != -1:
                neighbors.append([row-1,col])

        #check bottom neighbor
        if row + 1 < len(probs):
            if probs[row+1][col] != -1:
                neighbors.append([row+1,col])

        #check left neighbor
        if col - 1 >= 0:
            if probs[row][col-1] != -1:
                neighbors.append([row,col-1])

        #check right neighbor
        if col + 1 < len(probs[0]):
            if probs[row][col+1] != -1:
                neighbors.append([row,col+1])  

        return neighbors
    
    def truedistance(self, source, destination, probs):

        if self.probs[source[0]][source[1]] == -1 or self.probs[destination[0]][destination[1]] == -1:
            return -1

        queue = [[source,0]]
        seen = set()
        seen.add((source[0],source[1]))
        
        while queue:
            
            curr = queue.pop(0)
            # print(curr)
            if curr[0] == destination:
                return curr[1]
            
            neighbors = self.getNeighbors(curr[0],probs)

            for neighbor in neighbors:
                if (neighbor[0],neighbor[1]) not in seen:
                    queue.append([neighbor, curr[1]+1 ])
                    seen.add((neighbor[0],neighbor[1]))
        
    def getDistances(self):
        
        dummy = []

        #take source
        for i in range(len(self.probs)):
            temp = []
            for j in range(len(self.probs[0])):
                temp2 = []
                for p in range(len(self.probs)):
                    temp3 = []
                    for q in range(len(self.probs[0])):
                        temp3.append(self.truedistance([i,j],[p,q],self.probs))
                    temp2.append(temp3)
                temp.append(temp2)
            dummy.append(temp)
        return dummy

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
                                # if not self.distanceMatrix:
                                    
                                #     self.fourMatrix(self.probs) 
                                
                                manhattan = self.allPairDist[p][q][i][j]
                                # calculate manhattan distance  self.truedistance([i,j],[p,q],self.probs)
                                # manhattan = abs(p - i) + abs(q - j)

                                d = max(d, manhattan)

                maxdist = max(maxdist, d)

        ctr = 0
        for i in probs:
            for j in i:
                if j > 0:
                    ctr += 1
        
        #add weights to heuristic
        return ( 80 * (maxdist) + ctr ) * ((10 * ctr) ) * ( len(actions))

    def moveDrone(self, probs, action):
        
        if action == "U":
            newProbs = self.move_up(probs)
        elif action == "D":
            newProbs = self.move_down(probs)
        elif action == "L":
            newProbs = self.move_left(probs)
        elif action == "R":
            newProbs = self.move_right(probs)
        else:
            newProbs = copy.deepcopy(probs)
        return newProbs
        
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

    def getValidActions(self, sequence):
        
        if sequence and sequence[-1] == 'L':
            return ['U', 'L', 'D']
        elif sequence and sequence[-1] == 'U':
            return ['L', 'R', 'U']
        elif sequence and sequence[-1] == 'R':
            return  ['U', 'R', 'D']
        elif sequence and sequence[-1] == 'D':
            return ['D', 'R', 'L']
        else: 
            return ['L','U','R','D']

    def astar(self):

        heap = []
        validActions = ['L','U','R','D']
        probs = copy.deepcopy(self.probs)
        heuristic = self.compute_heuristic(probs,[])
        heap.append((heuristic, [], probs))
        visited = set()
        visited.add((tuple(map(tuple, probs))))

        while heap:
            currstate = heapq.heappop(heap)

            sequence = currstate[1]
            probs = currstate[2]

            if self.check(probs):
                return sequence

            validActions = self.getValidActions(sequence)
            for action in validActions:
                newprobs = self.moveDrone(probs, action)
                newheuristic = self.compute_heuristic(newprobs,sequence)
                if (tuple(map(tuple, newprobs))) not in visited:
                    visited.add((tuple(map(tuple, newprobs))))
                    heapq.heappush(heap, (newheuristic, sequence[:] + [action], newprobs))

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
        
        self.allPairDist = self.getDistances()

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
    