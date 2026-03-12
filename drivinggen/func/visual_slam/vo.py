

######3 vo fast ##############

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import os


def check_data(dataset_handler, outdir):
    image = dataset_handler.images[0]
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(image, cmap='gray')

    image_rgb = dataset_handler.images_rgb[0]
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(image_rgb)
    plt.savefig(os.path.join(outdir, 'rgb0.jpg'))

    depth = dataset_handler.depth_maps[0]
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(depth, cmap='jet')
    plt.savefig(os.path.join(outdir, 'depth0.jpg'))

    print("Depth map shape: {0}".format(depth.shape))
    v, u = depth.shape
    depth_val = depth[v-1, u-1]
    print("Depth value of the very bottom-right pixel of depth map {0} is {1:0.3f}".format(0, depth_val))


# 0319: update fron orb to sift

def extract_features(image, mask):
    """
    Find keypoints and descriptors for the image using SIFT

    Arguments:
    image -- a grayscale image
    mask -- optional mask to specify regions for feature detection

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    # 初始化 SIFT 检测器
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.015, edgeThreshold=7, sigma=1.3)

    
    # 检测关键点和计算描述符
    kp, des = sift.detectAndCompute(image, mask)
    
    return kp, des


def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(display)


def extract_features_dataset(images, masks):
    """
    Find keypoints and descriptors for each image in the dataset

    Arguments:
    images -- a list of grayscale images
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image

    Returns:
    kp_list -- a list of keypoints for each image in images
    des_list -- a list of descriptors for each image in images
    
    """
    kp_list = []
    des_list = []
    
    for img, mask in zip(images, masks):
        kp, des = extract_features(img, mask)
        kp_list.append(kp)
        des_list.append(des)
    
    return kp_list, des_list


def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    # Define FLANN parameters
    FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_LSH,
    #                     table_number = 6,
    #                     key_size = 12,
    #                     multi_probe_level = 1)
    # 0319: to tree for sift feature
    index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                        trees=6)
    # search_params = dict(checks = 50)
    search_params = dict(checks = 96)
    
    # Initiate FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find matches with FLANN
    try:
        match = flann.knnMatch(des1, des2, k=2)
        match2 = flann.knnMatch(des2, des1, k=2)
    except:
        print('matching fail')
        return [], []
    
    return match, match2


# Optional
def filter_matches_distance(match, dist_threshold, match2):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    for i, m in enumerate(match):
        if isinstance(m, tuple):
            if len(m) == 2:
                m, n = m
                if m.distance < (dist_threshold * n.distance):
                    # seems no improvement
                    # for j, m2 in enumerate(match2):
                    #     if len(m2) == 2:
                    #         m2, n2 = m2
                    #     else:
                    #         continue
                    #     if m.queryIdx == m2.trainIdx and m.trainIdx == m2.queryIdx:
                    #         filtered_match.append(m)
                    #         break
                    filtered_match.append(m)
            else:
                continue
    # filtered_match = [m for m, n in match if m.distance < (dist_threshold * n.distance)]
    
    return filtered_match


def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


def match_features_dataset(des_list):
    """
    Match features for each subsequent image pair in the dataset

    Arguments:
    des_list -- a list of descriptors for each image in the dataset
    match_features -- a function which maches features between a pair of images

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
               
    """
    matches = [match_features(des_list[i], des_list[i + 1]) for i in range((len(des_list) - 1))]
    matches1 = [match[0] for match in matches]
    matches2 = [match[1] for match in matches]
    
    return matches1, matches2


# Optional
def filter_matches_dataset(matches, dist_threshold, matches2):
    """
    Filter matched features by distance for each subsequent image pair in the dataset

    Arguments:
    filter_matches_distance -- a function which filters matched features from two images by distance between the best matches
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_matches -- list of good matches for each subsequent image pair in the dataset. 
                        Each matches[i] is a list of good matches, satisfying the distance threshold
               
    """
    filtered_matches = [filter_matches_distance(m, dist_threshold, m2) for m, m2 in zip(matches, matches2)]
    
    return filtered_matches


def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    image1 = image1.copy()
    image2 = image2.copy()
    
    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv2.circle(image1, p1, 1, (0, 255, 0), 1)
        # cv2.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv2.circle(image1, p2, 1, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv2.circle(image2, p2, 1, (255, 0, 0), 1)
    
    if is_show_img_after_move: 
        return image2
    else:
        return image1


import numpy as np

def is_pose_valid(rmat, tvec, det_tol=1e-2, trans_lim=50.0):
    """判断 rmat/tvec 是否数值正常、旋转正交且平移幅度合理"""
    if not (np.isfinite(rmat).all() and np.isfinite(tvec).all()):
        return False
    if abs(np.linalg.det(rmat) - 1.0) > det_tol:
        return False
    if np.linalg.norm(tvec) > trans_lim:
        return False
    return True


def safe_inv(mat4x4):
    """安全求逆，失败则返回 None"""
    try:
        inv = np.linalg.inv(mat4x4)
        if np.isfinite(inv).all():
            return inv
    except np.linalg.LinAlgError:
        pass
    return None



def estimate_motion(match, kp1, kp2, k, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix 
    
    Optional arguments:
    depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    
    objectpoints = []

    k = np.asarray(k, np.float64)
    # 只求一次 K^{-1}，失败则用伪逆兜底
    try:
        K_inv = np.linalg.inv(k)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(k)
    
    # Iterate through the matched features
    for m in match:
        # Get the pixel coordinates of features f[k - 1] and f[k]
        if isinstance(m, tuple):
            if len(m) == 2:
                m, n = m
            else:
                m = m[0]
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt
        
        # Get the scale of features f[k - 1] from the depth map
        s = depth1[int(v1), int(u1)]
        
        # Check for valid scale values
        # why 1000 is better for ego?
        if s < 80 and s > 1e-3:
            # Transform pixel coordinates to camera coordinates using the pinhole camera model
            p_c = K_inv @ (s * np.array([u1, v1, 1]))
            
            # Save the results
            image1_points.append([u1, v1])
            image2_points.append([u2, v2])
            objectpoints.append(p_c)
        
    if len(objectpoints) < 6:
        print("Not enough points for PnP RANSAC.")
        return rmat, tvec, image1_points, image2_points, False
        
    # Convert lists to numpy arrays
    objectpoints = np.vstack(objectpoints)
    imagepoints = np.array(image2_points)
    
    # Determine the camera pose from the Perspective-n-Point solution using the RANSAC scheme
    try:
        _, rvec, tvec, inliers = cv2.solvePnPRansac(objectpoints, imagepoints, k, None)
    except:
        print("PnP RANSAC failed.")
        return rmat, tvec, image1_points, image2_points, False
    # 使用内点进一步优化
    try:
        refined_objectPoints = objectpoints[inliers[:, 0]]
        refined_imagePoints = imagepoints[inliers[:, 0]]
        retval, rvec, tvec = cv2.solvePnP(refined_objectPoints, refined_imagePoints, k, None, rvec, tvec, useExtrinsicGuess=True)
    except:
        pass
    # doesn't work
    # retval, rvec, tvec = cv2.solvePnP(
    #     objectpoints,
    #     imagepoints,
    #     k,
    #     None,
    #     flags=cv2.SOLVEPNP_ITERATIVE
    # )

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    
    # ----------------- 新增：数值验证 -----------------
    ok_final = is_pose_valid(rmat, tvec)
    return rmat, tvec, image1_points, image2_points, ok_final


def estimate_trajectory(matches, kp_list, k, depth_maps=[], save='', dataset_handler=None):
    """
    Estimate complete camera trajectory from subsequent image pairs

    Arguments:
    estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    des_list -- a list of keypoints for each image in the dataset
    k -- camera calibration matrix 
    
    Optional arguments:
    depth_maps -- a list of depth maps for each frame. This argument is not needed if you use Essential Matrix Decomposition
    save -- a path to store camera movement images, it will not save images by default

    Returns:
    trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and   
                  trajectory[:, i] is a 3x1 numpy vector, such as:
                  
                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location
                  
                  * Consider that the origin of your trajectory cordinate system is located at the camera position 
                  when the first image (the one with index 0) was taken. The first camera location (index = 0) is geven 
                  at the initialization of this function

    """        
    # Create variables for computation

    trajectory = np.zeros((3, len(matches) + 1))
    robot_pose = np.zeros((len(matches) + 1, 4, 4))
    
    # Initialize camera pose
    robot_pose[0] = np.eye(4)

    pose_last = np.eye(4)  # Last pose for inertial extrapolation

    poses_3x3 = []
    ## init pose
    poses_3x3.append(
        (np.eye(3), np.array([0, 0, 0]))
    )
    
    # Iterate through the matched features
    for i in range(len(matches)):
        # Estimate camera motion between a pair of images
        rmat, tvec, image1_points, image2_points, ok = estimate_motion(matches[i], kp_list[i], kp_list[i + 1], k[i], depth_maps[i])
        
        # Save camera movement visualization
        if save:
            image = visualize_camera_movement(dataset_handler.images_rgb[i], image1_points, dataset_handler.images_rgb[i + 1], image2_points)
            plt.imsave('./{}/frame_{:05d}.jpg'.format(save, i), image)
        
        # Determine current pose from rotation and translation matrices
        if ok:
            current_pose = np.eye(4)
            current_pose[0:3, 0:3] = rmat
            current_pose[0:3, 3] = tvec.T
            pose_last = current_pose.copy()  # Update last pose for inertial extrapolation
        else:
            # current_pose = pose_last.copy()
            
            # ---- 只随机旋转；平移步长沿用上一次（Y 轴为上下） ----
            last_t = pose_last[:3, 3].astype(np.float64)   # 上一次相对位移向量
            L = float(np.linalg.norm(last_t))              # 步长（沿用）

            # 上一次的方向单位向量；若 L==0 给个占位方向（最终位移仍为 0）
            t_hat = (last_t / L) if L > 1e-12 else np.array([0.0, 0.0, 1.0])

            # 在“前向 180°（±90°）”内随机偏航角，避免后退
            max_yaw_deg = 90.0
            yaw = np.deg2rad(np.random.uniform(-max_yaw_deg, max_yaw_deg))

            # 绕 Y 轴的旋转矩阵（Y 为上下）
            c, s = np.cos(yaw), np.sin(yaw)
            Rrand = np.array([
                [ c, 0.0,  s],
                [0.0, 1.0, 0.0],
                [-s, 0.0,  c],
            ], dtype=np.float64)

            # 新的平移方向 = 旋转后的方向；步长不变
            t_new = (Rrand @ t_hat) * L

            current_pose = np.eye(4)
            current_pose[:3, :3] = Rrand
            current_pose[:3,  3] = t_new

            # 连续失败时每步都会重新随机朝向；步长仍沿用
            pose_last = current_pose.copy()
        
        # Build the robot's pose from the initial position by multiplying previous and current poses
        # robot_pose[i + 1] = robot_pose[i] @ np.linalg.inv(current_pose)
        # ③ 安全求逆；失败同样保持上一帧
        inv_pose = safe_inv(current_pose)
        if inv_pose is None:
            robot_pose[i+1] = robot_pose[i]
        else:
            robot_pose[i+1] = robot_pose[i] @ inv_pose

        poses_3x3.append(
            (robot_pose[i+1][:3,:3], robot_pose[i+1][:3,3])
        )
        
        # Calculate current camera position from origin
        position = robot_pose[i + 1] @ np.array([0., 0., 0., 1.])
        
        # Build trajectory
        trajectory[:, i + 1] = position[0:3]
        
    return trajectory, poses_3x3


# import cv2, numpy as np
# from typing import List, Sequence, Tuple

# # match, kp1, kp2, k, depth1=None
# # ----------------- 1. 单帧运动估计 -----------------
# def estimate_motion(
#     matches, kp1, kp2, K, depth1, depth_max = 80.0,
# ) -> Tuple[np.ndarray, np.ndarray, bool]:
#     """
#     返回 R (3×3), t (3×1), success
#     - 若成功:   success=True,  R/t 有效
#     - 若失败:   success=False, R=I, t=0 仅作占位
#     """
#     obj, img = [], []
#     dist_coeffs = np.array([-0.356123,0.172545,-0.00213,0.000464], dtype=np.float32)

#     if depth1 is not None:                # 用 PnP，需要 3D-2D 对
#         for m in matches:
#             m = m[0] if isinstance(m, tuple) else m
#             u1, v1 = kp1[m.queryIdx].pt
#             u2, v2 = kp2[m.trainIdx].pt
#             if (0 <= int(v1) < depth1.shape[0] and 0 <= int(u1) < depth1.shape[1]):
#                 s = float(depth1[int(v1), int(u1)])
#                 if 0 < s < depth_max:
#                     X = np.linalg.inv(K) @ (s * np.array([u1, v1, 1.0]))
#                     obj.append(X)
#                     img.append([u2, v2])

#     if len(obj) < 6:                      # 点太少 → 失败
#         print("Not enough points for PnP RANSAC.")
#         return np.eye(3), np.zeros((3, 1)), False

#     obj = np.float32(obj)
#     img = np.float32(img)
#     try:
#         ok, rvec, tvec, inl = cv2.solvePnPRansac(
#             obj, img, K, dist_coeffs,
#             flags=cv2.SOLVEPNP_ITERATIVE,
#             iterationsCount=300, reprojectionError=3.0, confidence=0.999
#         )
#     except:
#         print("PnP RANSAC failed.")
#         return np.eye(3), np.zeros((3, 1)), False

#     if not ok or inl is None or len(inl) < 6:
#         print("PnP RANSAC failed, not enough inliers.")
#         return np.eye(3), np.zeros((3, 1)), False

#     try:
#         cv2.solvePnP(obj[inl.ravel()], img[inl.ravel()], K, dist_coeffs,
#                     rvec, tvec, useExtrinsicGuess=True)
#     except:
#         print("PnP refinement failed.")
#         return np.eye(3), np.zeros((3, 1)), False

#     R, _ = cv2.Rodrigues(rvec)
#     return R, tvec, True

# # matches, kp_list, k, depth_maps=[], save='', dataset_handler=None
# # ----------------- 2. 轨迹估计，失败→惯性外推 -----------------
# def estimate_trajectory(
#     matches, kp_list, K, depth_maps, save='', dataset_handler=None,
# ) -> Tuple[np.ndarray, List[np.ndarray]]:
#     """
#     返回 轨迹 (3×N) 以及 每帧世界位姿 (4×4 list)
#     """
#     n = len(matches) + 1
#     traj   = np.zeros((3, n))
#     robot_pose = np.zeros((n, 4, 4))
#     robot_pose[0] = np.eye(4)                # 世界坐标下各帧位姿 T_w_i
#     ΔT_last = np.eye(4)                   # 上一次成功的相对变换
#     poses_3x3 = []
#     ## init pose
#     poses_3x3.append(
#         (np.eye(3), np.array([0, 0, 0]))
#     )

#     # K_seq   = K if isinstance(K, Sequence) else [K]*n
#     # depth_s = depth_maps if depth_maps is not None else [None]*n

#     for i in range(len(matches)):
#         R, t, ok = estimate_motion(
#             matches[i],
#             kp_list[i], kp_list[i+1],
#             K[i], depth_maps[i]
#         )

#         if ok:
#             # 本帧相对变换
#             ΔT = np.eye(4)
#             ΔT[:3,:3] = R
#             ΔT[:3, 3] = t.ravel()
#             ΔT_last   = ΔT.copy()         # 更新惯性
#         else:
#             ΔT = ΔT_last                  # 惯性外推

#         # 累乘得到世界坐标 pose_{i+1}
#         robot_pose[i + 1] = robot_pose[i] @ np.linalg.inv(ΔT)
#         poses_3x3.append(
#             (robot_pose[i+1][:3,:3], robot_pose[i+1][:3,3])
#         )
        
#         # Calculate current camera position from origin
#         position = robot_pose[i + 1] @ np.array([0., 0., 0., 1.])
        
#         # Build trajectory
#         traj[:, i + 1] = position[0:3]

#     return traj, poses_3x3
