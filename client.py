import zmq
import numpy as np
import pickle5 as pickle
from test import *
import matplotlib.pyplot as plt
# zmq client
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:5555")

def send_array(socket, array, flags=0, protocol=-1):
    #serialize numpy array
    data = pickle.dumps(array)
    socket.send(data)

# send a message
#while True:
#send_array(socket, np.array([[0,1],[2,3],[4,5]]))
    #wait one second
    #wait = socket.poll(timeout=1000)

def parameterize_ellipse(t,a,b,t0):
    return a*np.cos(t/t0),b*np.sin(t/t0)

t = 0
t0=100
a=100
b=200

true_xs=[]
true_ys=[]
points_orig = generate_points()

while t < 10000:
    t += 1
    x,y = parameterize_ellipse(t,a,b,t0)
    true_xs.append(x)
    true_ys.append(y)
    points = transform(points_orig,0,x,y)+np.random.normal(0, 20, points_orig.shape)
    send_array(socket, np.array([points,points_orig]))
    #wait one second
    #wait = socket.poll(timeout=1)

send_array(socket, "done")

plt.scatter(true_xs,true_ys,alpha=1.0)
plt.show()