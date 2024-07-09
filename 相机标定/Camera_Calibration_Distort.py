import cv2
import numpy as np

# 手动标记角点函数
def manual_corner_selection(event, x, y, flags, param):
    global corners, image_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append((x, y))
        cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Manual Corner Selection', image_copy)


image_path = '/home/lxy/frames/frame_0012.png'#图片路径
image = cv2.imread(image_path)

# 检查图像是否成功加载
if image is None:
    print(f"无法加载图像，请检查路径：{image_path}")
else:
    image_copy = image.copy()
    corners = []

    cv2.imshow('Manual Corner Selection', image)
    cv2.setMouseCallback('Manual Corner Selection', manual_corner_selection)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 转换角点为numpy数组
    corners = np.array(corners, dtype=np.float32)
    print("手动标记的角点：", corners)

    # 棋盘格为8x5，可以根据实际情况调整
    pattern_size = (8, 5)
    square_size = 27  # 每个方格的边长为27mm

    # 确保手动标记的角点数量与期望的数量一致
    if len(corners) == pattern_size[0] * pattern_size[1]:
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # 将手动标记的角点转换为适合相机标定的格式
        objpoints = [objp]
        imgpoints = [corners]

        # 使用手动标记的角点进行相机标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1::-1], None, None)

        if ret:
            print("相机标定成功。")
            print("相机矩阵：", mtx)
            print("畸变系数：", dist)

            # 保存相机矩阵和畸变系数到txt文件
            with open('/home/lxy/桌面/chessboard_27mm-master/result/camera_matrix.txt', 'w') as f:
                for line in mtx:
                    np.savetxt(f, line, fmt='%f')

            with open('/home/lxy/桌面/chessboard_27mm-master/result/dist_coeffs.txt', 'w') as f:
                np.savetxt(f, dist, fmt='%f')

            print("相机矩阵和畸变系数已保存到camera_matrix.txt和dist_coeffs.txt")
        else:
            print("相机标定失败。")
    else:
        print(f"手动标记的角点数量（{len(corners)}）与期望的数量（{pattern_size[0] * pattern_size[1]}）不匹配。")

