import cv2
import sys
import numpy as np
import glob

# 标定图像保存路径
photo_path = "./data/"


# 标定图像
def calibration_photo(photo_path):
    # 设置要标定的角点个数
    x_nums = 9  # x方向上的角点个数
    y_nums = 7
    # 设置(生成)标定图在世界坐标中的坐标
    world_point = np.zeros((x_nums * y_nums, 3), np.float32)  # 生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素
    world_point[:, :2] = np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)  # mgrid[]生成包含两个二维矩阵的矩阵，每个矩阵都有x_nums列,y_nums行
    # .T矩阵的转置
    # reshape()重新规划矩阵，但不改变矩阵元素
    # 保存角点坐标
    world_position = []
    image_position = []
    # 设置角点查找限制
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    # 获取所有标定图
    images = glob.glob(photo_path + '*.bmp')
    # print(images)
    for image_path in images:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 查找角点
        ok, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), None)

        if ok:
            # 把每一幅图像的世界坐标放到world_position中
            world_position.append(world_point)
            # 获取更精确的角点位置
            exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # 把获取的角点坐标放到image_position中
            image_position.append(exact_corners)
            # 可视化角点
            image = cv2.drawChessboardCorners(image,(x_nums,y_nums),exact_corners,ok)
            cv2.imshow('image_corner', image)
            cv2.waitKey(1000)
    # 计算内参数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_position, image_position, gray.shape[::-1], None, None)
    # 将内参保存起来
    np.savez('./data/intrinsic_parameters', mtx=mtx, dist=dist)
    print('mtx', mtx)
    print('--------------')
    print('dist', dist)
    # 计算偏差
    mean_error = 0
    for i in range(len(world_position)):
        image_position2, _ = cv2.projectPoints(world_position[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(image_position[i], image_position2, cv2.NORM_L2) / len(image_position2)
        mean_error += error
    print("total error: ", mean_error / len(image_position))


if __name__ == '__main__':
    calibration_photo(photo_path)