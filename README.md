# Ghosts In The Maze
Project with a maze environment with moving ghosts in which the agent has to navigate to the goal.

## Environment Setup
The environment for this problem is a maze-like square grid. Some of the cells are open (unblocked) and some are obstructed (blocked). An agent in the environment can move among and occupy the unblocked cells, but cannot enter the blocked cells.

We want to generate many environments to test our agent in, so to do so we will generate these mazes randomly:
starting with an empty 51 x 51 square grid, iterate through each cell, and with probability 0.28 make it blocked, with probability 0.72 make it unblocked.

However, since these mazes are generated randomly, they may not be very good (since blocks are distributed randomly, it’s entirely possible every cell is blocked, for instance). So we need to check the quality of the maze, and reject ones that are too blocked. Remove any blocks present from the upper left and lower right corners, and make sure there is a path from one to the other (moving in up/down/left/right directions)

## Agent
The agent is going to start in the upper left corner, and attempt to navigate to the lower right corner. The agent can move in the cardinal directions (up/down/left/right), but only between unblocked squares, and cannot move outside the 51x51 grid. At any time, the agent can ‘see’ the entirety of the maze, and use this information to plan a path.

## The Problem: Ghosts
The maze is full of ghosts. Each ghosts starts at a random location in the maze that is reachable from the upper left corner (so that no ghost gets walled off). If the agent enters a cell with a ghost (or a ghost enters the agent’s cell), the agent dies. This is to be avoided.

Each time the agent moves, the ghosts will also move. This means that whatever plan the agent initially generated to traverse the maze may at any point become blocked or invalid. This may mean the agent needs to re-plan its path through the maze to try to avoid the ghosts, and may have to repeatedly re-plan as the ghosts move.

At every timestep, a ghost picks one of its neighbors (up/down/left/right); if the picked neighbor is unblocked, the ghost moves to that cell; if the picked neighbor is blocked, the ghost either stays in place with probability 0.5, or moves into the blocked cell with probability 0.5. 

Every time the agent moves, the ghosts move according to the above rule. If the agent touches a ghost, the agent dies.

## Strategies To Solve The Environment
The different agents which use different strategies to solve/navigate the maze to win the game.

#### AGENT 1:
Agent 1 plans a the shortest path through the maze and executes it, ignoring the ghosts. This agent is incredibly efficient - it only has to plan a path once - but it makes no adjustments or updates due to a changing environment.

#### AGENT 2:
Agent 2 re-plans. At every timestep, Agent 2 recalculates a new path to the goal based on the current information, and executes the next step in this new path. Agent 2 is constantly updating and readjusting based on new information about the ghosts. Agent 2 doesn't make projections about the future. If all paths to goal are blocked, agent 2 attempts to move away from the nearest ghost.

#### AGENT 3:
Agent 3 forecasts. At every timestep, Agent 3 considers each possible move it might take (including staying in place), and ‘simulates’ the future based on the rules of Agent 2 past that point. For each possible move, this future is simulated some number of times, and then Agent 3 chooses among the moves with greatest success rates in these simulations. Agent 3 can be thought of as Agent 2, plus the ability to imagine the future.

#### AGENT 4:
Agent 4 is a free strategy agent which balances intelligence and efficiency.

#### LESS INFORMATION AGENTS:
Assuming that the agent loses sight of ghosts when they are in the walls, and cannot make decisions based on the location of these ghosts. How does this affect the performance of each agent?


