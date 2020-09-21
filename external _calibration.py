import cv2
import numpy as np
import math

# 读取相机内参
with np.load('./data/intrinsic_parameters.npz') as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

# 标定图像保存路径
photo_path = "./data/7.bmp"

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

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
    # 设置世界坐标的坐标
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    # 设置角点查找限制
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    image = cv2.imread(photo_path)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 查找角点
    ok, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), )
    if ok:
        # 获取更精确的角点位置
        exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 获取外参
        _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)

        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        _res_rvec = np.mat(rvec)

        print('matrix', mtx)
        _res_tvec = np.mat(tvec)
        # 旋转向量转旋转矩阵
        RotationMatrix= cv2.Rodrigues(_res_rvec)
        print('---------------------------------------')

        print('RotationMatrix:', RotationMatrix[0])
        euler = rotationMatrixToEulerAngles(RotationMatrix[0])
        print('---------------------------------------')
        print('euler', euler)
        print('---------------------------------------')

        print('平移向量', tvec)
        hom_mtx = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0 ,0.0 ,0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

        hom_mtx[0][3] = tvec[0]
        hom_mtx[1][3] = tvec[1]
        hom_mtx[2][3] = tvec[2]

        hom_mtx[0][0] = RotationMatrix[0][0][0]
        hom_mtx[0][1] = RotationMatrix[0][0][1]
        hom_mtx[0][2] = RotationMatrix[0][0][2]
        hom_mtx[1][0] = RotationMatrix[0][1][0]
        hom_mtx[1][1] = RotationMatrix[0][1][1]
        hom_mtx[1][2] = RotationMatrix[0][1][2]
        hom_mtx[2][0] = RotationMatrix[0][2][0]
        hom_mtx[2][1] = RotationMatrix[0][2][1]
        hom_mtx[2][2] = RotationMatrix[0][2][2]

        print('hom_mtx:', hom_mtx)

        # 可视化角点
        img = draw(image, corners, imgpts)
        # cv2.imshow('img', img)
        # cv2.waitKey(3000)
    else:
        print('Can`t find any corners')


if __name__ == '__main__':
    calibration_photo(photo_path)
    # cv2.waitKey()