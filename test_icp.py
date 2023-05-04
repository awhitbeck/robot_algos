from simpleicp import PointCloud, SimpleICP
import numpy as np
import random
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

# Read point clouds from xyz files into n-by-3 numpy arrays
#generate 85 random points in 3D
points = np.array([])
for i in range(85):
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    points = np.append(points,[a, b])
X_fix = points.reshape(85,2)
X_mov = transform(X_fix,0,100,200)#+np.random.normal(0, 0.1, X_fix.shape)
#add column of zeros to X_fix and X_mov
X_fix = np.append(X_fix,np.zeros((X_fix.shape[0],1)),axis=1)
X_mov = np.append(X_mov,np.zeros((X_mov.shape[0],1)),axis=1)
print(X_fix.shape)
print(X_mov.shape)
# Create point cloud objects
pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])

# Create simpleICP object, add point clouds, and run algorithm!
icp = SimpleICP()
icp.add_point_clouds(pc_fix, pc_mov)
H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=30)