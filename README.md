# EANN-racing

### Idea
In this simulation 2D cars are learning to complete the track using evolutionary neural networks.

### Implementation
Simulation implemented with pygame module. For deep learning part pytorch module was used.

### Neural Network
Each cars is runned by separate neural network, which is used as "engine". 
This neural network takes 5 distances from car to track borders as an input and gives "steering" parameter as an output (which changes the direction of the car from -15 to 15 degrees).
Architecture includes 2 hidden layer followed by ReLU activation, and output layer followed by Tanh.


### Algorithm
50 cars are spawned at each iteration. User chooses cars (any amount >=1, usually 2-4), which show good results in track completion. 
These cars are used as parents for next generation. The weights of all parents are averaged("crossbreedeing"). 
Then random normal noise (sigma = 0.25) is added independently for each car of next generation("mutation"). 


### Interface
LMB -- choosing car as parent
N -- new epoch (at least 1 car should be chosen)
S -- save current model to file
L -- load model from file
