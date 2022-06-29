# Learning-Mario-Agent
![alt text](https://miro.medium.com/max/1002/1*7TLBg5I9DSrvVwebZoA6JQ.gif)

# Preprocess the data:
1. Applying wrappers to environment: 
## What are Wrappers? 
Wrappers around functions are also knows as decorators which are a very powerful and useful tool in Python since it allows programmers to modify the behavior of function or class. Decorators allow us to wrap another function in order to extend the behavior of the wrapped function, without permanently modifying it.
```python
def func2(func):
    def ref():
        print("sentence 1")
        func()
        print("sentence 3")
        
    return ref    
def func():
    print("sentence 2")
    
func = func2(func)
func()

Output: 
sentence 1
sentence 2
sentence 3
```
```diff
1. GrayScaleObservation: Transform an RGB image to grayscale. By doing so reduces the size of the state representation without losing useful information.
2. ResizeObservation: Downsampling each observation into a square image.
3. SkipFrame: Consecutive frames don’t vary much, we can skip n-intermediate frames without losing much information. The n-th frame aggregates rewards accumulated over each skipped frame.
3. FrameStack: Then we squash consecutive frames of the environment into a single observation point to feed to our learning model. This way, we can identify if Mario was landing or jumping based on the direction of his movement in the previous several frames.
```
After applying the wrappers, we get the final wrapped state consisting of 4 gray-scaled consecutive frames stacked together. Each time Mario makes an action, the environment responds with a state of this structure. The structure is represented by a 3-D array of size [4, 84, 84]

# Agent:
## Mario is our agent who takes decisions in the Super Mario Environment based on rewards and punishments after every action. 
## There are three actions of an agent:
```diff
- Act according to the optimal action policy based on the current state (of the environment).
+ Remember experiences. Experience = (current state, current action, reward, next state). Mario caches and later recalls his experiences to update his action policy.
! Learn a better action policy over time

```
# Act
For a state, an agent can choose to Explore or Exploit. 
Explore: take a random action
Exploit: choose the most optimal action 
We start with a high value of exploration and decrease the exploration rate with increasing time steps.\
Limiting the action space to:\
- 0: walk right
- 1: jump right
# Remember/Memory
For memory, we create two functions, 
- cache(): Each time Mario performs an action, he stores the experience to his memory. His experience includes the current state, action performed, reward from the action, the next state, and whether the game is done.

- recall(): Mario randomly samples a batch of experiences from his memory, and uses that to learn the game.

# Learn
The Reinforcement Learning Algorithm that our Mario Agent uses is the Double Deep Q-Learning Network Algorithm. DDQN uses two ConvNets - Qonline and Qtarget that independently approximate the optimal action-value function. 
```
Mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  ```
# Calculating the TD Target and TD Estimate

![Alt Text](https://github.com/iiShreya/Learning-Mario-Agent/blob/master/images/tdtarget_tdestimate.png)

# Updating the Model
![Alt Text](https://github.com/iiShreya/Learning-Mario-Agent/blob/master/images/update.png)
# Metrics
- episode rewards
- episode lengths
- episode average loss 
- episode average Q values
# Print
- Episode
- Step
- Epsilon
- Mean Reward
- Mean Length
- Mean Loss
- Mean Q Value
- Time Delta 
- Time
[reference](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html#conclusion)

