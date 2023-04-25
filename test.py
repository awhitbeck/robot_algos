import random
import matplotlib.pyplot as plt
import numpy as np

# generate 85 pairs of random numbers
def generate_points():
    points = np.array([])
    for i in range(85):
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
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

def iterative_closest_point(source, target, max_iterations=20, tolerance=0.001):
    # initialize the transformation to the identity
    transformation = np.eye(3)

    # compute the source centroid
    source_centroid = np.mean(source, axis=0)

    # compute the target centroid
    target_centroid = np.mean(target, axis=0)

    # subtract the source centroid from the source
    source = source - source_centroid

    # subtract the target centroid from the target
    target = target - target_centroid

    # compute the covariance matrix
    covariance = np.dot(np.transpose(source), target)

    # compute the SVD of the covariance matrix
    U, S, V = np.linalg.svd(covariance)

    # compute the rotation
    rotation = np.dot(U, V)

    # special reflection case
    if np.linalg.det(rotation) < 0:
        V[2,:] *= -1
        rotation = np.dot(U, V)

    # compute the translation
    translation = target_centroid - np.dot(rotation, source_centroid)

    # update the transformation
    transformation[:2,:2] = rotation
    transformation[:2,2] = translation

    # compute the RMS error
    error = np.sqrt(np.sum(np.square(np.dot(source, rotation) + translation - target)) / source.shape[0])

    # return the transformation and error
    return transformation, error

def test():
    err=[]
    pull=np.array([])

    shiftx=-2000
    shifty=100
    ntrials=1000
    for i in range(ntrials):
        points = generate_points()

        #delete x% of points
        x=0.5
        points = np.delete(points, np.random.choice(points.shape[0], int(points.shape[0]*x), replace=False), axis=0)

        #add randome noise to points
        points_ = transform(points,0.,shiftx,shifty)
        points_ = points_ + np.random.normal(0, 400, points.shape)
        trans_, err_ = iterative_closest_point(points,points_)
        pull =  np.append(pull,[trans_[0,2],trans_[1,2]])
        err.append(err_)

    pull = pull.reshape(ntrials,2)
    print(pull)
    print(np.mean(err),np.std(err))
    plt.hist(err)
    #plt.scatter(points[:,0],points[:,1])
    #plt.scatter(points_[:,0],points_[:,1])
    plt.show()

    plt.scatter(pull[:,0],pull[:,1])
    plt.show()

"""
move to next feature
take image
get features
if first attempt, all features are added to  feature list
else,
    use ICP to match features to feature list and get transformation
    if transformation is too large, 
        skip this feature
    else
        use transformation to put current list of features in the same frame as the feature list
        check if any features are within a certain distance of the feature list
        if no, add to feature list and move to next feature
"""
