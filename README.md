# Learning-Mario-Agent
![alt text](https://miro.medium.com/max/1002/1*7TLBg5I9DSrvVwebZoA6JQ.gif)

# Preprocess the data:
1. Applying wrappers to environment: 
## Wrappers: 
Wrappers around the functions are also knows as decorators which are a very powerful and useful tool in Python since it allows programmers to modify the behavior of function or class. Decorators allow us to wrap another function in order to extend the behavior of the wrapped function, without permanently modifying it.
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
#number of episodes : 100
#learning increases with increasing number of episodes
reference: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html#conclusion

