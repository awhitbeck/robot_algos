import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

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

def make_correspondence(reference_points, query_points, max_distance=100):
    ## this function assumes that the mean of reference_points and query_points is zero

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)
    distances, indices = nbrs.kneighbors(query_points)
    ref=[]
    que=[]
    for i in range(len(distances)):
        if distances[i][0] < 100:
            #print("reference_points: ",reference_points[indices[i]][0])
            #print("query_points: ",query_points[i])

            ref.append(reference_points[indices[i]][0])
            que.append(query_points[i])
    return np.array(ref),np.array(que)

def iterative_closest_point(source, target, max_iterations=20, tolerance=0.001):

    err = 99999999999999.
    iteration = 0
    #print("source: ",source)
    #print("target: ",target)
    rotation = [[1.,0.],[0.,1.]]
    translation = [0.,0.]
    mean_source = np.mean(source, axis=0)
    mean_target = np.mean(target, axis=0)
    source = source - mean_source
    target = target - mean_target
    while iteration < max_iterations :
        # find correspondences between source and target
        target_corr,source_corr = make_correspondence(target,source)

        t,err_ = rigid_body_transformation(source_corr,target_corr)
        rotation = t[:2,:2]
        translation = t[:2,2]
        if abs(err - err_)/(err+err_) < tolerance :
            return rotation,translation+mean_source-mean_target,err_
        else :
            err = err_
            source = np.dot(source,rotation) + translation
            iteration += 1
            #print("iteration: ",iteration)
            #print("error: ",err_)
            #print("translation: ",translation+mean_source-mean_target)
            #print("rotation: ",rotation)

    return rotation, translation+mean_source-mean_target, err
def rigid_body_transformation(source, target):
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

    shiftx=-100
    shifty=300
    ntrials=200
    #delete x% of points
    x=0.5
    for i in range(ntrials):
        points = generate_points()
        #print("first point: ",points[0])
        #print("number of points: ",points.shape[0])
        #add randome noise to points
        points_ = transform(points,0.,shiftx,shifty)
        #print("first point_: ",points_[0])
        points_ = np.delete(points_, np.random.choice(points_.shape[0], int(points_.shape[0]*x), replace=False), axis=0)
        #print("first point_: ",points_[0])
        points_ = points_ + np.random.normal(0, 50, points_.shape)
        #print("number of points_: ",points_.shape[0])
        rot_, trans_, err_ = iterative_closest_point(points,points_)
        pull =  np.append(pull,[trans_[0],trans_[1]])
        err.append(err_)

    pull = pull.reshape(ntrials,2)
    #print(pull)
    #print(np.mean(err),np.std(err))
    plt.hist(err)
    plt.show()
    plt.scatter(pull[:,0],pull[:,1],alpha=0.1)
    #plt.scatter(points_[:,0],points_[:,1])
    plt.show()

    #plt.scatter(pull[:,0],pull[:,1])
    #plt.show()

##run test function when this file is run
if __name__ == "__main__":
    test()

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
