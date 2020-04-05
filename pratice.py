# Artificial intelligence for Business
# Optimizing warehouse flows with q-learning 
import numpy as np

# Parameters: gamma (facteur de réduction) & alpha (taux d'apprentissage)
gamma = 0.75
alpha = 0.9

# PART I - DEFINE THE ENVIRONMENT
#States
location_to_state = {k:v for k,v in zip(list('ABCDEFGHIJKL'),list(range(12)))}
state_to_location = {v:k for k,v in location_to_state.items()}

# PART II - BUILDING AI SOLUTION WITH Q-LEARNING
def training_q(R):    
    #Initialisation des Q-values
    Q = np.zeros([12, 12])
    for _ in range(1000):
        current_state = np.random.randint(0,12)
        playable_actions = []
        for j in range(12):
            if R[current_state,j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R[current_state, next_state] + gamma \
        * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state,next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
    return Q

        

# PART III - GOING INTO PRODUCTION
def route(starting_location, ending_location):
    #Rewardss
    R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
                  [1,0,1,0,0,1,0,0,0,0,0,0],
                  [0,1,0,0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,1,0,0,0],
                  [0,1,0,0,0,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,1,0,0,0,0],
                  [0,0,0,1,0,0,1,0,0,0,0,1],
                  [0,0,0,0,1,0,0,0,0,1,0,0],
                  [0,0,0,0,0,1,0,0,1,0,1,0],
                  [0,0,0,0,0,0,0,0,0,1,0,1],
                  [0,0,0,0,0,0,0,1,0,0,1,0]])
    ending_state = location_to_state[ending_location]
    R[ending_state, ending_state] = 1000
    Q = training_q(R)
    
    route = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

def route_with_intermediate(starting_location, ending_location, intermediate_location):
    return route(starting_location, intermediate_location)[:-1] \
    + route(intermediate_location, ending_location)
    
    
route("A", "B")
route_with_intermediate("A","G", "K")

        
        
    