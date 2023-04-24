import zmq
from test import *
import pickle5 as pickle
import numpy as np

# initialize a zmq socket
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:5555")

import pickle5 as pickle
def recv_array(socket):
    #deserialize numpy array
    data = socket.recv()
    array = pickle.loads(data)
    return array

xs=[]
ys=[]

while True:
    data = recv_array(socket)
    if data == "done":
        break
    points = data[0]
    points_orig = data[1]
    trans,err = iterative_closest_point(points,points_orig)
    xs.append(-trans[0,2])
    ys.append(-trans[1,2])


plt.scatter(xs,ys)
plt.show()