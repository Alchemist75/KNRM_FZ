import numpy as np
import random
def forward(input, target):
    expinput = np.exp(input) 
    #print (target * np.log(expinput / np.sum(expinput,axis=1)[:, None]))
    #x = raw_input()
    return -np.mean(np.sum(target * np.log(expinput / np.sum(expinput,axis=2)[:,:, None]),axis=2))
    pass

for i in range(100):
    i1 = np.ndarray([100,10,2])
    i2 = np.zeros([100,10,2])
    i2[:,:,1] = 1
    for x in range(2):
        for y in range(100):
            for z in range(10):
                i1[y][z][x] = random.random()
    print forward(i1,i2)

