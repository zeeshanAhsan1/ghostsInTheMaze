from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

start = (0,0)
destination = (50,50)
solution = {}
solution_bfs = {}


def getWalls():
    prob=0.28
    num_zeros=int(prob*100)
    num_ones=100-num_zeros
    mylist=[0]*num_zeros
    mylist+=[1]*num_ones
    random_index=random.randint(0,len(mylist)-1)
    return(mylist[random_index])

def getMaze():
    maze = np.ones((51,51),dtype=(int))
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            maze[i][j] = getWalls()
    return(maze)

def getGhosts(n):

    ghosts = {}
    for i in range(n):
        x = random.randint(0,50)
        y = random.randint(0,50)
        while((x,y) == (0,0) or (x,y) == (50,50)):  # get random (x,y) till (0,0) or (50,50) -> so start and end are not ghosts
            x = random.randint(0,50)
            y = random.randint(0,50)
        ghosts[i+1]=(x,y)   # Assign ghost with key 1 to N and value the co-ordinates in 2D grid
    return ghosts   #   Return ghost Dictionary

def ghostMovement(ghostDict,maze):

    for k,v in ghostDict.items():
        (x,y) = v
        if((x,y) == (50,50)): # in end cell
            r = random.choice([2,3])
        elif((x,y) == (0,0)): # in start cell
            r = random.choice([1,4])
        elif((x,y) == (50,0)): # last row - 1st column
            r = random.choice([1,3])
        elif((x,y) == (0,50)): # first row - last column
            r = random.choice([2,4])
        elif(x != 0 and y == 50): # not top row, in last column
            r = random.choice([2,3,4])
        elif(x != 0 and y == 0): # not top row, in 1st column
            r = random.choice([1,3,4])
        elif(y != 0 and x == 0): # not 1st column, in 1st row
            r = random.choice([1,2,4])
        elif(y != 0 and x == 50): #not 1st column, in last row
            r = random.choice([1,2,3])
        else:
            r = random.choice([1,2,3,4])
        (a,b) = v
        if(r == 1): # Move Right
            if(maze[a][b+1] == 0): # walled cell
                ch = random.choice([1,2])
                if(ch == 1): # Move to the walled cell
                    b = b+1
                else:   # Don't move
                    b = b
            else:   # Not walled Cell, Move with normal Probability
                b = b + 1
            ghostDict[k] = (a,b)
        if(r == 2): # Move Left
            if(maze[a][b-1] == 0): # Walled cell
                ch = random.choice([1,2])
                if(ch == 1): # Move to the walled cell
                    b = b-1
                else:   # Don't move
                    b = b
            else:   # Not a walled cell
                b=b-1
            ghostDict[k] = (a,b)
        if(r == 3): # Move Up
            if(maze[a-1][b] == 0): # Walled cell
                ch = random.choice([1,2])
                if(ch == 1): # Move to the walled cell
                    a = a-1
                else:   # Don't Move
                    a = a
            else:   # Not walled cell
                a = a-1
            ghostDict[k] = (a,b)
        if(r == 4): # Move Down
            if(maze[a+1][b] == 0):  # Walled cell
                ch = random.choice([1,2])
                if(ch == 1): # Move to the walled Cell
                    a = a+1
                else:   # Don't Move
                    a = a
            else:   # Not a walled cell
                a = a + 1
            ghostDict[k] = (a,b)
    return(ghostDict)
    
def dfsSearch(a,start):
    stack = []
    visited = []
    x,y = start[0],start[1]
    stack.append((x,y))
    while len(stack) > 0:   # Iterate till stack is empty
        current = (x,y)
        if(current == (50,50)):
            return True
        
        if(y-1 >=0 and a[x][y-1] != 0 and (x,y-1) not in visited): #Path available on left side
           cellleft = (x,y-1)
           stack.append(cellleft)
           
        if(y+1 <= 50 and a[x][y+1] != 0 and (x,y+1) not in visited): #Path available on right side
            cellright = (x,y+1)
            stack.append(cellright)
            
        if(x-1 >= 0 and a[x-1][y] !=0 and (x-1,y) not in visited): #Path available on Up side
            cellup = (x-1,y)
            stack.append(cellup)
            
        if(x+1 <= 50 and a[x+1][y] != 0 and (x+1,y) not in visited): #Path available on Down Side
            celldown = (x+1,y)
            stack.append(celldown)
            
        x,y = stack.pop()
        visited.append(current)

def checkValid(maze):
    if(maze[0][0] == 1 and maze[50][50] == 1):
        #print("Start and Finish unblocked")
        if(dfsSearch(maze,start) == True):
            #print("Valid Maze Found")
            return True
        else:
            #print("No Valid Path from Start to Finish")
            return False
    else:
        #print("Start and Finish Blocked")
        return False


def getValidMaze():
    while(True):
        maze = getMaze()
        if checkValid(maze):
            return(maze)

def nearestGhost(ghostSet,curCell):
    minDist = np.inf
    nearGhost = (1000,10000)
    for x in ghostSet:
        x_dis = abs(x[0] - curCell[0])  # Calculate X distance from ghost
        y_dis = abs(x[1] - curCell[1])  # Calculate X distance from ghost
        distance = x_dis + y_dis    # Calculate Manhattan Distance from the nearest ghost
        if(distance < minDist):
            minDist = distance
            nearGhost = x   # Update Nearest ghost
    return nearGhost    #   Return the Nearest ghost


def djikstra(maze,start,ghostDict):

    unvisited = {}
    curCell = start
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            unvisited[(i,j)] = float('inf') #Instantiate the unvisted dictionary for all keys to infinity cost
    
    unvisited[start] = 0    # Set the start cell as zero cost
    visited = {}    # Make an empty visited dictionary
    revPath = {}    # Make an empty reverse path Dictionary

    curCell = min(unvisited, key=unvisited.get) # Assign the current cell as the lowest cost cell frem the dictionary of unvisited cells
    #print(curCell)

    ghostSet = set()
    for k,v in ghostDict.items():   # Make the set of ghosts from ghost dictionary
        ghostSet.add(v)

    while unvisited:    # Traverse till unvisited dictionary gets empty
        curCell = min(unvisited, key=unvisited.get) # Assign the current cell as the lowest cost cell frem the dictionary of unvisited cells
        x,y = curCell[0],curCell[1] # Take the x and y cio-ordinates of the current cell
        visited[curCell] = unvisited[curCell] # Add the Value of Unvisited dictionary for current cell to the corresponding key in visited dictionary
        if(curCell == (50,50)): # If we reach (50,50) goal cell
            if((50,50) in revPath.keys()):  # If (50,50) is in keys of the reverse path dictionary
                path = buildPath(revPath,start) # Then build a path
                pathList = []   # Make an empty path 
                for key,val in path.items():
                    pathList.append(val)    # Make an array of cells which only contain reverse path
                return (pathList)
            else:
                return(-2)
            #return(50,50)
        #Check Neighbors of the current Cell

        neighborList = []
        if(y+1 <= 50 and maze[x][y+1] !=0 and (x,y+1) not in ghostSet): #Rneighbor
            neighbor = (x,y+1)
            neighborList.append(neighbor)
            #print("curCell :" ,neighbor)
        if(y-1 >= 0 and maze[x][y-1] !=0 and (x,y-1) not in ghostSet): #Lneigh
            neighbor = (x,y-1)
            neighborList.append(neighbor)
            #print("curCell :" ,neighbor)
        if(x+1 <= 50 and maze[x+1][y] !=0 and (x+1,y) not in ghostSet): #UNeigh
            neighbor = (x+1,y)
            neighborList.append(neighbor)
            #print("curCell :" ,neighbor)
        if(x-1 >= 0 and maze[x-1][y] != 0 and (x-1,y) not in ghostSet): #DNeigh
            neighbor = (x-1,y)
            neighborList.append(neighbor)
            #print("curCell :" ,neighbor)


        #Check if Neighbor is already present in visited dict
        for nbr in neighborList:
            if(nbr in visited): # For each valid neighbor if it ios already visited then do nothing and continue to the next neighbor
                continue    
            tempDist = unvisited[curCell] + 1   
            if(tempDist < unvisited[nbr]):
                unvisited[nbr] = tempDist   # Update the distance of neighbor cell if the current distance is less than previous distance in the dictionary for this cell
                revPath[nbr] = curCell  # Set the reverse path key of neighbor cell to current cell value

        unvisited.pop(curCell)  # Remove the current cell from visited cell dictionary

def buildPath(revPath,start):
    path = {}   # Empty path dictionary
    end = (50,50)

    x,y = end[0],end[1]

    while ((x,y) != (start[0],start[1])):   # Traverse the dictionary till we reach the start cell from goal cell
        path[revPath[(x,y)]] = (x,y)    #   Make the path dict with key as Reverse path Value for that key
        (x,y) = revPath[(x,y)]      # Update the co-ordinates

    return path     # Return the path dictionary

def djikstra_ag2(maze, start, ghostDict):

    #Build ghostSet
    ghostSet = set()
    for key,val in ghostDict.items():
        ghostSet.add(val)

    curCell = start[0],start[1]

    if(curCell in ghostSet):
        return(-2)
    
    while (curCell != (50,50)):

        djPath = djikstra(maze, curCell, ghostDict) # Calculate the path
        while(djPath == -2):    # While no path to destination is found

            # Get nearest ghost Location
            nearGhost = nearestGhost(ghostSet, curCell)
            # Build neighbors List
            neighborList = []
            x,y = curCell[0],curCell[1]
            if(y+1 <= 50 and maze[x][y+1] !=0 and (x,y+1) not in ghostSet): #Rneighbor
                neighbor = (x,y+1)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
            if(y-1 >= 0 and maze[x][y-1] !=0 and (x,y-1) not in ghostSet): #Lneigh
                neighbor = (x,y-1)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
            if(x+1 <= 50 and maze[x+1][y] !=0 and (x+1,y) not in ghostSet): #UNeigh
                neighbor = (x+1,y)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
            if(x-1 >= 0 and maze[x-1][y] != 0 and (x-1,y) not in ghostSet): #DNeigh
                neighbor = (x-1,y)
                neighborList.append(neighbor)

                # if(len(neighborList) == 0):
                #     return(-2) #No moves possible. Agent is blocked from all sides
            
            while(len(neighborList) == 0): #If All neighbor cells are blocked
                #Move the ghosts
                ghostDict = ghostMovement(ghostDict, maze)
                ghostSet.clear()
                for key,val in ghostDict.items():
                    ghostSet.add(val)

                if(curCell in ghostSet): #If ghost Kills agent return -2
                    return(-2)
                else:
                    neighborList.clear() # Otherwise hope for the ghost to move away from the neighbor cell 
                    x,y = curCell[0],curCell[1] # and try calculating the neighbors again

                    #I do this process till I get a Valid neighbor cell or the ghost kills agent

                    if(y+1 <= 50 and maze[x][y+1] !=0 and (x,y+1) not in ghostSet): #Rneighbor
                        neighbor = (x,y+1)
                        neighborList.append(neighbor)
                        #print("curCell :" ,neighbor)
                    if(y-1 >= 0 and maze[x][y-1] !=0 and (x,y-1) not in ghostSet): #Lneigh
                        neighbor = (x,y-1)
                        neighborList.append(neighbor)
                        #print("curCell :" ,neighbor)
                    if(x+1 <= 50 and maze[x+1][y] !=0 and (x+1,y) not in ghostSet): #UNeigh
                        neighbor = (x+1,y)
                        neighborList.append(neighbor)
                        #print("curCell :" ,neighbor)
                    if(x-1 >= 0 and maze[x-1][y] != 0 and (x-1,y) not in ghostSet): #DNeigh
                        neighbor = (x-1,y)
                        neighborList.append(neighbor)


            max_dist = float('inf')
            
            # Check distance of nearest ghost from all valid neighbor Cells and 
            # take the move with most distance from nearest ghost
            for neigh in neighborList:
                x_dis = abs(neigh[0] - nearGhost[0])
                y_dis = abs(neigh[1] - nearGhost[1])
                dist = x_dis + y_dis
                if(dist<max_dist):
                    max_dist = dist
                    curCell = neigh #Make the move

            #Move ghost and make new ghostSet
            ghostDict = ghostMovement(ghostDict, maze)
            ghostSet.clear()
            for key,val in ghostDict.items():
                ghostSet.add(val)

            if(curCell in ghostSet): # Agent killed then return -2
                return(-2)

            # Calculate a new Path
            djPath = djikstra(maze, curCell, ghostDict)
            
        
        curCell = djPath.pop()
        #print("Cur Cell :" , curCell)

        ghostDict = ghostMovement(ghostDict, maze) # Move the ghosts and update the ghost dictionary
        ghostSet.clear()
        for key,val in ghostDict.items(): # Update the ghost Set
            ghostSet.add(val)

        if(curCell in ghostSet):    # Check if agent died due to this movement of ghosts
            return(-2)


        if(curCell == (50,50)): #Check if agent reached the goal cell
            return(100)


def djikstra_ag3(maze,start,ghostDict):

    curCell = start
    sims = 8    # Number of simulations of future for each neighbor of current cell
    ghostSet = set()
    #Build GhostSet
    for key,val in ghostDict.items():
        ghostSet.add(val)

    x,y = curCell[0], curCell[1]
    rVal = 0
    dVal =0
    # Calculate the right neighbors
    if(maze[x][y+1] !=0 and (x,y+1) not in ghostSet): #Rneighbor
        neighbor = (x,y+1)
        for itr in range(sims):
            res = djikstra_ag2(maze, neighbor, ghostDict)
            if(res == 100):
                rVal += 1
        #print("curCell :" ,neighbor
    if(maze[x+1][y] != 0 and (x+1,y) not in ghostSet): #DNeigh
        neighbor = (x+1,y)
        for itr in range(sims):
            res = djikstra_ag2(maze, neighbor, ghostDict)
            if(res == 100):
                dVal += 1

    if (rVal == max(rVal,dVal)):
        curCell = (0,1)
    elif(dVal == max(rVal,dVal)):
        curCell = (1,0)

    #Move Ghosts
    ghostDict = ghostMovement(ghostDict, maze)
    ghostSet.clear()
    for key,val in ghostDict.items():
        ghostSet.add(val)
    
    if(curCell in ghostSet): # Check if ghost killed agent
        return(-2)
    #print("Cur Cell : ", curCell)
    while(curCell != (50,50)):

        #Calculate Neighbors
        rVal = 0
        dVal = 0
        uVal = 0
        lVal = 0
        cVal = 0
        x,y = curCell[0], curCell[1]

        #SImulate future from current cell
        for itr in range(sims):
            res = djikstra_ag2(maze, curCell, ghostDict)
            if(res == 100):
                cVal += 1

        #Simulate future from right neighbor
        if(y+1 <= 50 and maze[x][y+1] !=0 and (x,y+1) not in ghostSet): #Rneighbor
            neighbor = (x,y+1)
            for itr in range(sims):
                res = djikstra_ag2(maze, neighbor, ghostDict)
                if(res == 100):
                    rVal += 1
            # neighborList.append(neighbor)
            #print("curCell :" ,neighbor)
        #Simulate future from Left neighbor
        if(y-1 >= 0 and maze[x][y-1] !=0 and (x,y-1) not in ghostSet): #Lneigh
            neighbor = (x,y-1)
            for itr in range(sims):
                res = djikstra_ag2(maze, neighbor, ghostDict)
                if(res == 100):
                    lVal += 1
            # neighborList.append(neighbor)
            #print("curCell :" ,neighbor)
        #Simulate future from up neighbor
        if(x+1 <= 50 and maze[x+1][y] !=0 and (x+1,y) not in ghostSet): #DNeigh
            neighbor = (x+1,y)
            for itr in range(sims):
                res = djikstra_ag2(maze, neighbor, ghostDict)
                if(res == 100):
                    dVal += 1
            #neighborList.append(neighbor)
            #print("curCell :" ,neighbor)
        #Simulate future from down neighbor
        if(x-1 >= 0 and maze[x-1][y] != 0 and (x-1,y) not in ghostSet): #UNeigh
            neighbor = (x-1,y)
            for itr in range(sims):
                res = djikstra_ag2(maze, neighbor, ghostDict)
                if(res == 100):
                    uVal += 1
            # neighborList.append(neighbor)

        #Find maximum value of survivability from each move
        maxVal = max(rVal,dVal,lVal,uVal)
        print("RVAL :", rVal, " DVAL :", dVal, " UVAL :", uVal, " lVal :",lVal)

        if(maxVal == rVal):
            curCell = (curCell[0],curCell[1]+1) #Move right
        elif(maxVal == dVal):
            curCell = (curCell[0]+1,curCell[1]) #Move Down
        elif(maxVal == cVal):
            curCell = curCell   #Stay in Place
        elif(maxVal == uVal):
            curCell = (curCell[0]-1,curCell[1]) # Move up
        elif(maxVal == lVal):
            curCell = (curCell[0],curCell[1]-1) # Move Left

        print("CurCell : ", curCell)
        ghostDict = ghostMovement(ghostDict, maze)  # Move ghosts after this new movement

        ghostSet.clear()
        for key,val in ghostDict.items():   #Build a new ghost set with the above ghosts
            ghostSet.add(val)

        if(curCell in ghostSet):    # Check if agent has died
            return(-2)

        if(curCell == (50,50)):     # Check if agent reached the goal cell
            return (100)


def djikstra_ag4(maze,start,ghostDict):

    sim = 5
    #Build ghostSet
    ghostSet = set()
    for key,val in ghostDict.items():
        ghostSet.add(val)

    curCell = start[0],start[1]
    #print("Cur Cell :" , curCell)

    if(curCell in ghostSet):    # Check if ghost killed agent
        return(-2)
    
    while (curCell != (50,50)):     # Move till reaching goal cell

        #Calculate the best path to Goal with ghosts
        
        djPath = djikstra(maze, curCell, ghostDict)
        while(djPath == -2):    #If no path from current cell then
            #Build neighborList
            neighborList = []
            #Simulate agent2 and find the best neighbour to goto (Gaining Intelligence Step)
            x,y = curCell[0],curCell[1]
            rVal,uVal,dVal,lVal = 0, 0, 0, 0
            if(y+1 <= 50 and maze[x][y+1] !=0 and (x,y+1) not in ghostSet): #Rneighbor
                neighbor = (x,y+1)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
                #Simulate future from here
                for itr in range(sim):
                    res = djikstra_ag2(maze, (x,y+1), ghostDict)
                    if(res == 100):
                        rVal +=1
            if(y-1 >= 0 and maze[x][y-1] !=0 and (x,y-1) not in ghostSet): #Lneigh
                neighbor = (x,y-1)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
                #Simulate future from here
                for itr in range(sim):
                    res = djikstra_ag2(maze, (x,y-1), ghostDict)
                    if(res == 100):
                        lVal +=1
            if(x+1 <= 50 and maze[x+1][y] !=0 and (x+1,y) not in ghostSet): #DNeigh
                neighbor = (x+1,y)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
                #Simulate future from here
                for itr in range(sim):
                    res = djikstra_ag2(maze, (x+1,y), ghostDict)
                    if(res == 100):
                        dVal +=1
            if(x-1 >= 0 and maze[x-1][y] != 0 and (x-1,y) not in ghostSet): #UNeigh
                neighbor = (x-1,y)
                neighborList.append(neighbor)
                #Simulate future from here
                for itr in range(sim): 
                    res = djikstra_ag2(maze, (x-1,y), ghostDict)
                    if(res == 100):
                        uVal +=1

            maxVal = max(lVal,uVal,dVal,rVal)
            #If no movement is possible from neighbor cells then stay in place
            if(len(neighborList) == 0):
                curCell = curCell
            elif(maxVal == 0):  #If no success in any direction then move away from nearest ghost
                max_dist = float('inf')
                nearGhost = nearestGhost(ghostSet, curCell)
                for neigh in neighborList:
                    x_dis = abs(neigh[0] - nearGhost[0])
                    y_dis = abs(neigh[1] - nearGhost[1])
                    dist = x_dis + y_dis
                    if(dist<max_dist):
                        max_dist = dist
                        curCell = neigh # Move to the cell furthest away from nearest ghost

            # Make a move with highest survivability
            elif(dVal == maxVal):
                curCell = (x+1,y)
            elif(rVal == maxVal):
                curCell = (x,y+1)
            elif(lVal == maxVal):
                curCell = (x,y-1)
            elif(uVal == maxVal):
                curCell = (x-1,y)
            
            if(curCell == (50,50)):
                return(100)
            
            #Move Ghosts
            ghostDict = ghostMovement(ghostDict, maze)
            ghostSet.clear()
            for key,val in ghostDict.items():
                ghostSet.add(val)

            #Check if agent dead
            if(curCell in ghostSet):
                return(-2)
            djPath = djikstra(maze, curCell, ghostDict)
        
        curCell = djPath.pop()
        #print("Cur Cell :" , curCell)

        #Move Ghosts
        ghostDict = ghostMovement(ghostDict, maze)
        ghostSet.clear()
        for key,val in ghostDict.items():
            ghostSet.add(val)

        if(curCell == (50,50)):
            return(100)

        if(curCell in ghostSet):
            return(-2)

def less_info__ag4(maze,start,ghostDict):

    #Number of simulations when no path
    sim = 5

    #Build ghostSet
    ghostSet = set()
    for key,val in ghostDict.items():
        ghostSet.add(val)
    
    #agentGhostSet - The set of ghosts not in walls which agent can see

    agentGhostSet = set()
    #cntAgGhost = 0
    for gh in ghostSet:
        if(maze[gh[0]][gh[1]] != 0):
            agentGhostSet.add(gh)
            #cntAgGhost += 1

    #Make a new dictionary to pass as agent5 can only see the ghosts in the Path
    agentGhostDict = {}
    ctr = 0
    for gh in agentGhostSet:
        agentGhostDict[ctr+1] = gh



    curCell = start[0],start[1]
    print("Cur Cell :" , curCell)

    if(curCell in ghostSet): # Check if ghost killed ghost
        return(-2)
    
    while (curCell != (50,50)): #Traverse till we reach the goal cell

        #Calculate the best path to Goal with ghosts


        djPath = djikstra(maze, curCell, agentGhostDict)
        while(djPath == -2):    #If no path from current cell then
            #Build neighborList
            neighborList = []
            #Simulate agent2 and find the best neighbour to goto (Gaining Intelligence Step)
            x,y = curCell[0],curCell[1]
            rVal,uVal,dVal,lVal = 0, 0, 0, 0
            if(y+1 <= 50 and maze[x][y+1] !=0 and (x,y+1) not in agentGhostSet): #Rneighbor
                neighbor = (x,y+1)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
                #Simulate future from here
                for itr in range(sim):
                    res = djikstra_ag2(maze, (x,y+1), agentGhostDict)
                    if(res == 100):
                        rVal +=1
            if(y-1 >= 0 and maze[x][y-1] !=0 and (x,y-1) not in agentGhostSet): #Lneigh
                neighbor = (x,y-1)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
                #Simulate future from here
                for itr in range(sim):
                    res = djikstra_ag2(maze, (x,y-1), agentGhostDict)
                    if(res == 100):
                        lVal +=1
            if(x+1 <= 50 and maze[x+1][y] !=0 and (x+1,y) not in agentGhostSet): #DNeigh
                neighbor = (x+1,y)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
                #Simulate future from here
                for itr in range(sim):
                    res = djikstra_ag2(maze, (x+1,y), agentGhostDict)
                    if(res == 100):
                        dVal +=1
            if(x-1 >= 0 and maze[x-1][y] != 0 and (x-1,y) not in agentGhostSet): #UNeigh
                neighbor = (x-1,y)
                neighborList.append(neighbor)
                #Simulate future from here
                for itr in range(sim): 
                    res = djikstra_ag2(maze, (x-1,y+1), agentGhostDict)
                    if(res == 100):
                        uVal +=1

            maxVal = max(lVal,uVal,dVal,rVal)
            #If no movement is possible from neighbor cells then stay in place
            if(len(neighborList) == 0):
                curCell = curCell
            elif(maxVal == 0):  #If no success in any direction then stay in place and hope for the best
                curCell = curCell
            elif(dVal == maxVal):
                curCell = (x+1,y)
            elif(rVal == maxVal):
                curCell = (x,y+1)
            elif(lVal == maxVal):
                curCell = (x,y-1)
            elif(uVal == maxVal):
                curCell = (x-1,y)
            
            if(curCell == (50,50)):
                return(100)
            
            #Move Ghosts
            ghostDict = ghostMovement(ghostDict, maze)
            ghostSet.clear()
            for key,val in ghostDict.items():
                ghostSet.add(val)
            
            #Create New agentGhostSet and agentGhostDictionary
            agentGhostSet.clear()
            for gh in ghostSet:
                if(maze[gh[0]][gh[1]] != 0):
                    agentGhostSet.add(gh)

            ctr = 0
            agentGhostDict.clear()  # Build the new ghost set which only agent can see while planning a path
            for gh in agentGhostSet:
                agentGhostDict[ctr+1] = gh



            #Check if agent dead
            if(curCell in ghostSet):
                return(-2)
            djPath = djikstra(maze, curCell, agentGhostDict)    # Calculate a path to goal from current cell with the new ghost
                                                                #  dictionary with ghosts which only agent can see
        curCell = djPath.pop()
        print("Cur Cell :" , curCell)

        #Move Ghosts
        ghostDict = ghostMovement(ghostDict, maze)
        ghostSet.clear()
        for key,val in ghostDict.items():
            ghostSet.add(val)

        #Create New agentGhostSet and agentGhostDictionary
        agentGhostSet.clear()
        for gh in ghostSet:
            if(maze[gh[0]][gh[1]] != 0):
                agentGhostSet.add(gh)   # Calculate a new ghost Set which only agent can see

        agentGhostDict.clear()
        ctr = 0
        for gh in agentGhostSet:
            agentGhostDict[ctr+1] = gh  # Build a new dictionary of ghosts which only agent can see from previous step of ghost Set

        if(curCell == (50,50)): # Check if agent has reached goal cell
            return(100)

        if(curCell in ghostSet):    # Check if agent is killed by ghost
            return(-2)

def djikstra_ag1(maze,ghostDict):

    # As agent1 can't see the ghosts 
    # I use an empty dictionary for no ghosts to project the condition where he does not see the ghosts
    start = (0,0)
    end = (50,50)
    ghostSet = set()
    agentGhostDict ={}
    path = djikstra(maze, start, agentGhostDict)    # Path without ghosts
    
    curCell = start

    while(curCell != (50,50)):

        curCell = path.pop()    # Make a move in the calculated path
        ghostDict = ghostMovement(ghostDict, maze)  #move the ghosts 
        ghostSet.clear()
        for key,val in ghostDict.items(): # Build a ghost set 
            ghostSet.add(val)

        if(curCell in ghostSet):    # Check if ghost killed agent
            return(-2)
        if(curCell == end): # Check if agent reached goal cell
            return(100)

def less_info_ag1(maze,ghostDict):
    start = (0,0)
    end = (50,50)
    ghostSet = set()
    agentGhostDict ={}
    path = djikstra(maze, start, agentGhostDict)    # Path without ghosts
    
    curCell = start

    while(curCell != (50,50)):

        curCell = path.pop()
        ghostDict = ghostMovement(ghostDict, maze)
        ghostSet.clear()
        for key,val in ghostDict.items():
            ghostSet.add(val)

        if(curCell in ghostSet):
            return(-2)
        if(curCell == end):
            return(100)

def less_info_ag2(maze, start, ghostDict):

    #Build ghostSet
    ghostSet = set()
    for key,val in ghostDict.items():
        ghostSet.add(val)

    curCell = start[0],start[1]

    #Build agent ghost set - Set of ghosts not in walls which agent can see
    agentGhostSet = set()
    #cntAgGhost = 0
    for gh in ghostSet:
        if(maze[gh[0]][gh[1]] != 0):
            agentGhostSet.add(gh)


    #Make a new dictionary to pass as agent5 can only see the ghosts in the Path
    agentGhostDict = {}
    ctr = 0
    for gh in agentGhostSet:
        agentGhostDict[ctr+1] = gh



    if(curCell in ghostSet): # Check if ghost killed the agent
        return(-2)
    
    while (curCell != (50,50)): # Go till I reach the goal cell

        djPath = djikstra(maze, curCell, agentGhostDict) # calculate path with the ghosts which only agent can see
        while(djPath == -2):    # If no path to goal is possible

            # Get nearest ghost Location
            nearGhost = nearestGhost(agentGhostSet, curCell)
            # Build neighbors List
            neighborList = []
            x,y = curCell[0],curCell[1]
            if(y+1 <= 50 and maze[x][y+1] !=0 and (x,y+1) not in agentGhostSet): #Rneighbor
                neighbor = (x,y+1)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
            if(y-1 >= 0 and maze[x][y-1] !=0 and (x,y-1) not in agentGhostSet): #Lneigh
                neighbor = (x,y-1)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
            if(x+1 <= 50 and maze[x+1][y] !=0 and (x+1,y) not in agentGhostSet): #UNeigh
                neighbor = (x+1,y)
                neighborList.append(neighbor)
                #print("curCell :" ,neighbor)
            if(x-1 >= 0 and maze[x-1][y] != 0 and (x-1,y) not in agentGhostSet): #DNeigh
                neighbor = (x-1,y)
                neighborList.append(neighbor)

                # if(len(neighborList) == 0):
                #     return(-2) #No moves possible. Agent is blocked from all sides
            
            while(len(neighborList) == 0): #If All neighbor cells are blocked
                #Move the ghosts
                ghostDict = ghostMovement(ghostDict, maze)
                ghostSet.clear()
                for key,val in ghostDict.items():
                    ghostSet.add(val)

                if(curCell in ghostSet): #If ghost Kills agent return -2
                    return(-2)
                else:
                    neighborList.clear() # Otherwise hope for the ghost to move away from the neighbor cell 
                    x,y = curCell[0],curCell[1] # and try calculating the neighbors again

                    #I do this process till I get a Valid neighbor cell or the ghost kills agent

                    if(y+1 <= 50 and maze[x][y+1] !=0 and (x,y+1) not in ghostSet): #Rneighbor
                        neighbor = (x,y+1)
                        neighborList.append(neighbor)
                        #print("curCell :" ,neighbor)
                    if(y-1 >= 0 and maze[x][y-1] !=0 and (x,y-1) not in ghostSet): #Lneigh
                        neighbor = (x,y-1)
                        neighborList.append(neighbor)
                        #print("curCell :" ,neighbor)
                    if(x+1 <= 50 and maze[x+1][y] !=0 and (x+1,y) not in ghostSet): #UNeigh
                        neighbor = (x+1,y)
                        neighborList.append(neighbor)
                        #print("curCell :" ,neighbor)
                    if(x-1 >= 0 and maze[x-1][y] != 0 and (x-1,y) not in ghostSet): #DNeigh
                        neighbor = (x-1,y)
                        neighborList.append(neighbor)


            max_dist = float('inf')
            
            # Check distance of nearest ghost from all valid neighbor Cells and 
            # take the move with most distance from nearest ghost
            for neigh in neighborList:
                x_dis = abs(neigh[0] - nearGhost[0])
                y_dis = abs(neigh[1] - nearGhost[1])
                dist = x_dis + y_dis
                if(dist<max_dist):
                    max_dist = dist
                    curCell = neigh #Make the move

            #Move ghost and make new ghostSet
            ghostDict = ghostMovement(ghostDict, maze)
            ghostSet.clear()
            for key,val in ghostDict.items():
                ghostSet.add(val)

            # Make a new ghost set for lower info environment
            agentGhostSet.clear()
            for gh in ghostSet:
                if(maze[gh[0]][gh[1]] != 0):
                    agentGhostSet.add(gh)

            # Make the new agent ghost dictionary which agent can see
            ctr = 0
            agentGhostDict.clear()
            for gh in agentGhostSet:
                agentGhostDict[ctr+1] = gh
                
            

            if(curCell in ghostSet):
                return(-2)

            # Calculate a new Path
            djPath = djikstra(maze, curCell, agentGhostDict)
            
        
        curCell = djPath.pop()
        #print("Cur Cell :" , curCell)

        ghostDict = ghostMovement(ghostDict, maze)
        ghostSet.clear()
        for key,val in ghostDict.items():
            ghostSet.add(val)

        # Make a new ghost set for lower info environment
        agentGhostSet.clear()
        for gh in ghostSet:
            if(maze[gh[0]][gh[1]] != 0):
                agentGhostSet.add(gh)

        # Make the new agent ghost dictionary which agent can see
        ctr = 0
        agentGhostDict.clear()
        for gh in agentGhostSet:
            agentGhostDict[ctr+1] = gh

        if(curCell in ghostSet):
            return(-2)


        if(curCell == (50,50)):
            return(100) 

            


        
    

# djikstra_ag2(maze, (0,0), ghosts)

def test():
    
    sim = 150
    counter = 0
    for i in range(sim):
        maze = getValidMaze()
        ghosts = getGhosts(30)
        #res = djikstra_ag3(maze, (0,0), ghosts)
        #res = djikstra_ag2(maze, (0,0), ghosts)
        #res = less_info__ag4(maze, start, ghosts)
        #res = djikstra_ag4(maze, (0,0), ghosts)
        res = djikstra_ag1(maze, ghosts)
        print("Res = ", res)
        if(res == 100):
            counter += 1
        print("Current Sim: ", i+1, "Current Success % : ", (counter/(i+1) * 100))
    print("Success % = ", (counter/sim) * 100)


def agentsTest():
    k = 0
    gh = [100]
    success =[]
    sim = 150
    for x in gh:
        counter = 0
        for i in range(sim):
            maze = getValidMaze()
            ghostDict = getGhosts(x)
            res = less_info_ag2(maze, (0,0), ghostDict)
            #res = less_info__ag4(maze, start, ghostDict)
            #res = djikstra_ag1(maze, ghostDict)
            if(res == 100):
                counter +=1
            print("Counter/Success Value : ", counter)
            print("Current Ghost : ", x, "Current Sim: ", i+1, "Current Success % : ", (counter/(i+1) * 100))
        k = (counter/sim) * 100
        print("Final Success Value for no of ghost:",x,"is",k ,"%")
        success.append(k)
    return success

#test()

def graph_agent1():

    x = [30,35,40,45,50,55,60,65,70,75,80,85,90,95]
    y = [38.67,35.33,32.66,20.66,19.3,16.67,14,8.67,10,11.33,11.33,5.33,4.66,0]

    plt.plot(x,y)

    plt.xlabel('No. of ghosts for Agent 1')
    plt.ylabel('Survivability %')

    plt.title('Survivability graph averaged over 150 mazes')

    plt.show()


def graph_agent2():

    x = [30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,110,120]
    y = [50,46,33,36,28,24,22,18,19,19,3,7,5,6,7,4,0]

    plt.plot(x,y)

    plt.xlabel('No. of ghosts for Agent 2')
    plt.ylabel('Survivability %')

    plt.title('Survivability graph averaged over 100 mazes')

    plt.show()


def graph_agent4():

    x = [30,40,50,60,70,80,90,100,110,120]
    y = [54,40,32,26,17.4,8,8,6,4,0.8]

    plt.plot(x,y)

    plt.xlabel('No. of ghosts for Agent 4')
    plt.ylabel('Survivability %')

    plt.title('Survivability graph averaged over 50 mazes')

    plt.show()


def graph_agent2_blind():

    x = [30,35,40,45,50,55,60,70,80,90,100]
    y = [43,41,23,25,19,9,26,10,4,7,0.67]

    plt.plot(x,y)

    plt.xlabel("No. of ghosts for Agent 2 which can't see ghosts in walls")
    plt.ylabel('Survivability %')

    plt.title('Survivability graph averaged over 50 mazes')

    plt.show()



def graph_agent4_blind():

    x = [30,35,40,45,50,55,60,70,80,90,100]
    y = [34,38,36,16,12,22,14,12,10,6,2]

    plt.plot(x,y)

    plt.xlabel("No. of ghosts for Agent 4 which can't see ghosts in walls")
    plt.ylabel('Survivability %')

    plt.title('Survivability graph averaged over 50 mazes')

    plt.show()



# successArray = agentsTest()
# print(successArray)

#graph_agent4_blind()

