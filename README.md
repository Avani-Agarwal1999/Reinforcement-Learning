# Reinforcement-Learning
In this project we take a sidewalk of size (W x L) where the
width and length of the sidewalk is 6 and 25. We initialize 8 to have
two rows representing paths that are not sidewalks. The position
of the agent can be described by the x and y co-ordinates of the
sidewalk. The agent starts from the top left side of the sidewalk
and must move towards the right hand side.
The agent is not allowed to go backwards. The agent may move
up, down, left , right or sideways and by doing so it may encounter
litter for which it is rewarded or an obstacle for which it is penalized.
The objective is to reach the end of the side walk. All the cells in
the last column that is (𝑖, 24) where i is in range (1, 6) means agent
has reached the end of the sidewalk. There are 4 modules that are
implemented in this assignment. Later all four modules will be
linearly combined. The four modules are as following:
(1) Sidewalk Module: Staying on the sidewalk
(2) Obstacle Module:Avoiding obstacles (negative reward)
(3) Litter Module:Collecting litter along the way (positive reward)
(4) Forth Module: Reaching the end of the sidewalk
