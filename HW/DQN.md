# Q-learning
## terminology
agent, state, action, reward(reward value, reward matrix **R**), Q matrix(states x actions), episode
## steps:
1. Set the gamma hyperparameter, and environment rewards in **R**.
2. Initializing matrix Q
3. For each episode:
   - Select a random initial state 
   - Do while the goal state has not been reached:
       i. Select an action
       ii. Taking this possible action, consider going to the next state
       iii. Get maximum Q for this next state based on all possible actions.
       iv. Q(state,action)=R(state,action)+Gamma\*Max(Q(next state,all actions))
       v. Set the next state as the current state
