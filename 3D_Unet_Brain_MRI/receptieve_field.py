import numpy as np


convs = [{'f':3, 's':1}, {'f':3, 's':1}, {'f':3, 's':1}, {'f':3, 's':1}]
def rf(d):
    l = [1]
    for i in range(len(convs)):
        si = np.prod([x['s'] for x in convs[:i]])
        li = l[i] + (convs[i]['f'] - 1) * si
        l.append(li)
        print("The receptive field of layer %d is %d" %(i+1, li))