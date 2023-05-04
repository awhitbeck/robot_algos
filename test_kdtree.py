from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import matplotlib.pyplot as plt
def generate_points():
    points = np.array([])
    for i in range(85):
        a = random.randint(-1000, 1000)
        b = random.randint(-1000, 1000)
        points = np.append(points,[a, b])
    points = points.reshape(85,2)
    return points

def transform(points, theta=0., tx=0., ty=0., scale=1.):
    # transpose points
    #points = np.transpose(points)

    # rotate points by theta
    points = [[ np.cos(theta)*p[0]+np.sin(theta)*p[1], -np.sin(theta)*p[0]+np.cos(theta)*p[1] ]  for p in points ]

    # scale points by scale
    points = [[ scale*p[0], scale*p[1] ] for p in points ]

    # translate points by tx and ty
    points = [[ p[0]+tx, p[1]+ty ] for p in points ]

    return np.array(points)


def main():

    ## generate random points
    reference_points = generate_points()
    ## transform points b
    query_points = transform(reference_points,0,10,10)+np.random.normal(0, 20, reference_points.shape)

    # subtract mean from query_points and reference_points
    query_points = query_points - np.mean(query_points, axis=0)
    reference_points = reference_points - np.mean(reference_points, axis=0)
    print('mean query_points: ',np.mean(query_points, axis=0))
    print('mean reference_points: ',np.mean(reference_points, axis=0))

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)
    distances, indices = nbrs.kneighbors(query_points)
    dx=[]
    dy=[]
    for i in range(len(distances)):
        if distances[i][0] < 100:
            print("reference_points: ",reference_points[indices[i]][0])
            print("query_points: ",query_points[i])
            dx.append(reference_points[indices[i]][0][0]-query_points[i][0])
            dy.append(reference_points[indices[i]][0][1]-query_points[i][1])
    plt.scatter(dx,dy)
    plt.show()

if __name__ == "__main__":
    main()