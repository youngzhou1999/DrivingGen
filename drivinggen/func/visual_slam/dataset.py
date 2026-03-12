import os

import math
import numpy as np
import cv2 as cv

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


class DatasetHandler:

    def __init__(self, rgb_np_list, depth_np_list, k):
        # Define number of frames
        self.num_frames = len(rgb_np_list)

        # Set up paths
        # root_dir_path = os.path.dirname(os.path.realpath(__file__))
        # self.image_dir = os.path.join(root_dir_path, 'data/rgb')
        # self.depth_dir = os.path.join(root_dir_path, 'data/depth')

        # Set up data holders
        self.images = []
        self.images_rgb = rgb_np_list
        self.depth_maps = depth_np_list

        # self.k = np.array([[640, 0, 640],
        #                    [0, 480, 480],
        #                    [0,   0,   1]], dtype=np.float32)

        self.k = k
        

        # Read first frame
        # self.read_frame()
        for img in rgb_np_list:
            self.images.append(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
        
    def read_frame(self):
        self._read_depth()
        self._read_image()
              
    def _read_image(self):
        for i in range(1, self.num_frames + 1):
            zeroes = "0" * (5 - len(str(i)))
            im_name = "{0}/frame_{1}{2}.png".format(self.image_dir, zeroes, str(i))
            self.images.append(cv.imread(im_name, flags=0))
            self.images_rgb.append(cv.imread(im_name)[:, :, ::-1])
            print ("Data loading: {0}%".format(int((i + self.num_frames) / (self.num_frames * 2 - 1) * 100)), end="\r")
            
       
    def _read_depth(self):
        for i in range(1, self.num_frames + 1):
            zeroes = "0" * (5 - len(str(i)))
            depth_name = "{0}/frame_{1}{2}.dat".format(self.depth_dir, zeroes, str(i))
            depth = np.loadtxt(
                depth_name,
                delimiter=',',
                dtype=np.float64) * 1000.0
            self.depth_maps.append(depth)
            print ("Data loading: {0}%".format(int(i / (self.num_frames * 2 - 1) * 100)), end="\r")
            
        
def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    image1 = image1.copy()
    image2 = image2.copy()
    
    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv.circle(image1, p1, 5, (0, 255, 0), 1)
        cv.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv.circle(image2, p2, 5, (255, 0, 0), 1)
    
    if is_show_img_after_move: 
        return image2
    else:
        return image1


def estimate_yaw_from_xy(xs, ys):
    xys = np.stack([xs, ys], axis=-1)
    deltas = np.diff(xys, axis=0)
    yaws = np.arctan2(deltas[:, 1], deltas[:, 0])
    yaws = np.append(yaws, yaws[-1])  # 最后一帧复制前一帧方向
    return yaws


from matplotlib.patches import Polygon

def draw_car(ax, xs, ys, length=4.5, width=2.0, color='red', zorder=1):
    yaws = estimate_yaw_from_xy(xs, ys)
    for x, y, yaw in zip(xs, ys, yaws):
        # 定义车身局部坐标的四个角点（以中心为原点）
        corners = np.array([
            [ length/2,  width/2],
            [ length/2, -width/2],
            [-length/2, -width/2],
            [-length/2,  width/2],
        ])

        # 构造旋转矩阵（yaw 是弧度）
        rot = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])

        # 坐标旋转 + 平移
        transformed = corners @ rot.T + np.array([x, y])

        # 创建矩形 Patch
        patch = Polygon(transformed, closed=True, edgecolor='none', facecolor=color, alpha=0.3, zorder=1)
        ax.add_patch(patch)

        # 可选：画朝向线（车头）
        # head = np.array([length/2, 0]) @ rot.T + np.array([x, y])
        # ax.plot([x, head[0]], [y, head[1]], color=color, linewidth=1.5, zorder=zorder)


def visualize_trajectory(trajectory, outdir, gt=None, others=None, others_gt=None, map_scale=1, car_length=4.5, car_width=2.0, draw_polygon=0):
    # Unpack X Y Z each trajectory point
    locX = []
    locY = []
    locZ = []
    # This values are required for keeping equal scale on each plot.
    # matplotlib equal axis may be somewhat confusing in some situations because of its various scale on
    # different axis on multiple plots
    # max = -math.inf
    # min = math.inf

    # Needed for better visualisation
    maxY = -math.inf
    minY = math.inf

    for i in range(0, trajectory.shape[1]):
        current_pos = trajectory[:, i]
        
        locX.append(current_pos.item(0))   # sync carla
        locY.append(current_pos.item(1))
        locZ.append(current_pos.item(2))
        # if np.amax(current_pos) > max:
        #     max = np.amax(current_pos)
        # if np.amin(current_pos) < min:
        #     min = np.amin(current_pos)

        if current_pos.item(1) > maxY:
            maxY = current_pos.item(1)
        if current_pos.item(1) < minY:
            minY = current_pos.item(1)

    auxY_line = locY[0] + locY[-1]
    # if max > 0 and min > 0:
    #     minY = auxY_line - (max - min) / 2
    #     maxY = auxY_line + (max - min) / 2
    # elif max < 0 and min < 0:
    #     minY = auxY_line + (min - max) / 2
    #     maxY = auxY_line - (min - max) / 2
    # else:
    #     minY = auxY_line - (max - min) / 2
    #     maxY = auxY_line + (max - min) / 2

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot the figure
    # fig = plt.figure(figsize=(8, 6), dpi=100)
    # fig, traj_main_plt = plt.subplots(figsize=(8, 6), dpi=100)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi=100)

    traj_main_plt = axes[0,0]

    gt_plt = axes[0,1]

    ego_plt = axes[1,0]
    others_plt = axes[1,1]

    gspec = gridspec.GridSpec(1, 1)
    # ZY_plt = plt.subplot(gspec[0, 1:])
    # YX_plt = plt.subplot(gspec[1:, 0])
    # traj_main_plt = plt.subplot(gspec[0:, 0:])
    # D3_plt = plt.subplot(gspec[0, 0], projection='3d')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Actual trajectory plotting ZX
    toffset = 1.06
    traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=toffset)
    traj_main_plt.set_title("Trajectory (Z, X)", y=1)
    if draw_polygon:
        draw_car(traj_main_plt, locX, locZ, car_length, car_width, color=colors[0], zorder=1)
        draw_car(ego_plt, locX, locZ, car_length, car_width, color=colors[0], zorder=1)
    
    traj_main_plt.plot(locX, locZ, ".-", label="Ego", zorder=6, linewidth=1, markersize=4, color=colors[0])
    ego_plt.plot(locX, locZ, ".-", label="Ego", zorder=6, linewidth=1, markersize=4, color=colors[0])

    max_x, min_x = np.max(locX), np.min(locX)
    max_y, min_y = np.max(locZ), np.min(locZ)
    if gt is not None:
        if draw_polygon:
            draw_car(traj_main_plt, gt[:,0], gt[:,1], car_length, car_width, color=colors[1], zorder=1)
            draw_car(gt_plt, gt[:,0], gt[:,1], car_length, car_width, color=colors[1], zorder=1)
        
        traj_main_plt.plot(gt[:,0], gt[:,1], ".-", label="EgoGT", zorder=1, linewidth=1, markersize=4, color=colors[1])
        gt_plt.plot(gt[:,0], gt[:,1], ".-", label="EgoGT", zorder=1, linewidth=1, markersize=4, color=colors[1])
        # min = np.min(gt) if np.min(gt) < min else min
        # max = np.max(gt) if max < np.max(gt) else max
        max_x = max(max_x, np.max(gt[:,0]))
        min_x = min(min_x, np.min(gt[:,0]))
        max_y = max(max_y, np.max(gt[:,1]))
        min_y = min(min_y, np.min(gt[:,1]))
    if others is not None:
        # traj_main_plt.scatter(others[:,0], others[:,1], label="Others", c='grey', s=4)
        # 1->0 or 0->1
        try:
            others, others_w_ego, others_w_opt = others
        except:
            others_w_ego = None
            others_w_opt = None
        if draw_polygon:
            draw_car(traj_main_plt, others[:,0], others[:,1], car_length, car_width, color=colors[2], zorder=1)
            draw_car(others_plt, others[:,0], others[:,1], car_length, car_width, color=colors[2], zorder=1)
        
        traj_main_plt.plot(others[:,0], others[:,1], ".-", label="Focal", zorder=3, linewidth=1, markersize=4, color=colors[2])
        others_plt.plot(others[:,0], others[:,1], ".-", label="Focal", zorder=3, linewidth=1, markersize=4, color=colors[2])

        max_x = max(max_x, np.max(others[:,0]))
        min_x = min(min_x, np.min(others[:,0]))
        max_y = max(max_y, np.max(others[:,1]))
        min_y = min(min_y, np.min(others[:,1]))
        if others_w_ego is not None:
            # traj_main_plt.plot(others_w_ego[:,0], others_w_ego[:,1], ".-", label="Focal_w_ego", zorder=1, linewidth=1, markersize=4)
            max_x = max(max_x, np.max(others_w_ego[:,0]))
            min_x = min(min_x, np.min(others_w_ego[:,0]))
            max_y = max(max_y, np.max(others_w_ego[:,1]))
            min_y = min(min_y, np.min(others_w_ego[:,1]))
        if others_w_opt is not None:
            if draw_polygon:
                draw_car(traj_main_plt, others_w_opt[:,0], others_w_opt[:,1], car_length, car_width, color=colors[3], zorder=1)
                draw_car(others_plt, others_w_opt[:,0], others_w_opt[:,1], car_length, car_width, color=colors[3], zorder=1)
            traj_main_plt.plot(others_w_opt[:,0], others_w_opt[:,1], ".-", label="Focal_w_opt", zorder=6, linewidth=1, markersize=4, color=colors[3])

            others_plt.plot(others_w_opt[:,0], others_w_opt[:,1], ".-", label="Focal_w_opt", zorder=6, linewidth=1, markersize=4, color=colors[3])
            max_x = max(max_x, np.max(others_w_opt[:,0]))
            min_x = min(min_x, np.min(others_w_opt[:,0]))
            max_y = max(max_y, np.max(others_w_opt[:,1]))
            min_y = min(min_y, np.min(others_w_opt[:,1]))
    if others_gt is not None:
        count = 0
        for o_gt in others_gt:
            if abs(o_gt[-1, 0] - o_gt[0, 0]) > 10:
                count += 1
                if count == 2:
                    print(f'len: {o_gt.shape[0]}')
                    if draw_polygon:
                        draw_car(traj_main_plt, o_gt[:,1], o_gt[:,0], car_length, car_width, color=colors[4], zorder=1)
                        draw_car(gt_plt, o_gt[:,1], o_gt[:,0], car_length, car_width, color=colors[4], zorder=1)
                    
                    traj_main_plt.plot(o_gt[:,1], o_gt[:,0], ".-", label="FocalGT", zorder=2, linewidth=1, markersize=4, color=colors[4])
                    gt_plt.plot(o_gt[:,1], o_gt[:,0], ".-", label="FocalGT", zorder=2, linewidth=1, markersize=4, color=colors[4])
                    max_x = max(max_x, np.max(o_gt[:,1]))
                    min_x = min(min_x, np.min(o_gt[:,1]))
                    max_y = max(max_y, np.max(o_gt[:,0]))
                    min_y = min(min_y, np.min(o_gt[:,0]))
    def set_plot(this_plot):
        this_plot.set_xlabel("X")
        this_plot.set_ylabel("Z")
        # traj_main_plt.axes.yaxis.set_ticklabels([])
        # Plot reference lines
        # traj_main_plt.plot([locZ[0], locZ[-1]], [locX[0], locX[-1]], "--", label="Auxiliary line", zorder=0, linewidth=1)
        # Plot camera initial location
        # traj_main_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
        # this_plot.set_xlim([-25*map_scale, 25*map_scale])
        # this_plot.set_ylim([-3*map_scale, 47*map_scale])
        this_plot.set_xlim([min_x-3, max_x+3])
        this_plot.set_ylim([-3, max_y+3])
        this_plot.legend(loc=1, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    set_plot(traj_main_plt)
    set_plot(gt_plt)
    set_plot(ego_plt)
    set_plot(others_plt)

    # # Plot ZY
    # # ZY_plt.set_title("Z Y", y=toffset)
    # ZY_plt.set_ylabel("Y", labelpad=-4)
    # ZY_plt.axes.xaxis.set_ticklabels([])
    # ZY_plt.plot(locZ, locY, ".-", linewidth=1, markersize=4, zorder=0)
    # ZY_plt.plot([locZ[0], locZ[-1]], [(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], "--", linewidth=1, zorder=1)
    # ZY_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    # ZY_plt.set_xlim([min, max])
    # ZY_plt.set_ylim([minY, maxY])

    # # Plot YX
    # # YX_plt.set_title("Y X", y=toffset)
    # YX_plt.set_ylabel("X")
    # YX_plt.set_xlabel("Y")
    # YX_plt.plot(locY, locX, ".-", linewidth=1, markersize=4, zorder=0)
    # YX_plt.plot([(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], [locX[0], locX[-1]], "--", linewidth=1, zorder=1)
    # YX_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    # YX_plt.set_xlim([minY, maxY])
    # YX_plt.set_ylim([min, max])

    # # Plot 3D
    # D3_plt.set_title("3D trajectory", y=toffset)
    # D3_plt.plot3D(locX, locZ, locY, zorder=0)
    # D3_plt.scatter(0, 0, 0, s=8, c="red", zorder=1)
    # D3_plt.set_xlim3d(min, max)
    # D3_plt.set_ylim3d(min, max)
    # D3_plt.set_zlim3d(min, max)
    # D3_plt.tick_params(direction='out', pad=-2)
    # D3_plt.set_xlabel("X", labelpad=0)
    # D3_plt.set_ylabel("Z", labelpad=0)
    # D3_plt.set_zlabel("Y", labelpad=-2)
    
    # # plt.axis('equal')
    # D3_plt.view_init(45, azim=30)
    # plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(outdir, 'pose.jpg'))