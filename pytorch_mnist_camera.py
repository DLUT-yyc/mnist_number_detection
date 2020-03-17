
import torch
from pytorch_mnist import LeNet
import torchvision as tv
import torchvision.transforms as transforms
import cv2
import numpy as np

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion


model_name = 'model/net_012.pth'
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载LeNet在mnist数据集上训练得到的模型
def load_mnist_model():
    # 定义网络模型LeNet
    net = LeNet().to(device)
    # 加载硬盘上的参数文件
    checkpoint = torch.load(model_name)
    # 加载参数到网络
    net.load_state_dict(checkpoint)

    return net


# 将4个点组成的轮廓曲线画出
def draw_approx_curve(img, approx):
    for i in range(len(approx) - 1):
        cv2.line(img, (approx[i,0,0], approx[i,0,1]), (approx[i+1,0,0], approx[i+1,0,1]), (0, 0, 255), 2)
    cv2.line(img, (approx[0,0,0], approx[0,0,1]), (approx[-1,0,0], approx[-1,0,1]), (0, 0, 255), 2)


# 计算两个点之间的欧式距离
def dis_points(p1, p2):
    p1 = np.squeeze(p1)
    p2 = np.squeeze(p2)
    dist = np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))
    return dist


# 对图像中的4个点进行排序，顺序为左上，左下，右下，右上
def sort4points(points):
    lmin = 1e6
    lmax = 0
    imin = 0
    imax = 0
    for i in range(4):
        if points[i, 0] + points[i, 1] < lmin:
            lmin = points[i, 0] + points[i, 1]
            imin = i
        if points[i, 0] + points[i, 1] > lmax:
            lmax = points[i, 0] + points[i, 1]
            imax = i
    lx = 1e6
    ix = 0
    for i in range(4):
        if i != imin and i != imax:
            if points[i, 0] < lx:
                lx = points[i, 0]
                ix = i
    for i in range(4):
        if i != imin and i != imax and i != ix:
            iy = i
    newpts = np.zeros_like(points)
    newpts[0] = points[imin]
    newpts[1] = points[iy]
    newpts[2] = points[imax]
    newpts[3] = points[ix]
    return newpts


# 提取图像中的方框，并进行手写数字识别
def box_extractor(img, net):
    # 图像边缘提取，使用Canny边缘检测算法
    edges = cv2.Canny(img, 100, 200)
    # 在边缘中查找图像中的封闭轮廓
    cnts, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 下面这行代码可以将封闭轮廓画出，用于调试
    # img = cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
    cx, cy = 0, 0
    # 遍历每条封闭轮廓
    for cnt in cnts:
        # 求轮廓周长
        peri = cv2.arcLength(cnt, True)
        # 对轮廓进行多边形拟合
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # 找到4边型
        if len(approx) == 4 and cv2.isContourConvex(approx):
            edges = np.zeros(4, np.float32)
            # 计算4个边的边长
            edges[0] = dis_points(approx[0], approx[1])
            edges[1] = dis_points(approx[1], approx[2])
            edges[2] = dis_points(approx[2], approx[3])
            edges[3] = dis_points(approx[3], approx[0])
            # 计算4个边的最小值，进制，标准差
            e_min = np.min(edges)
            e_avg = np.mean(edges)
            e_std = np.std(edges)

            # 对4边形 进行约束
            if e_min > 10 and e_std / e_avg < 0.2:
                draw_approx_curve(img, approx)

                cx = (approx[0,0,0] + approx[1,0,0] + approx[2,0,0] + approx[3,0,0]) / 4.
                cy = (approx[0,0,1] + approx[1,0,1] + approx[2,0,1] + approx[3,0,1]) / 4.
                # 目标框在垂直正视视角的4个点的坐标
                pts_res = np.float32([[0, 0], [28, 0], [28, 28], [0, 28]])
                approx = np.squeeze(approx).astype(np.float32)
                # 调整四边形4个点的顺序，为左上，左下，右下，右上
                approx = sort4points(approx)
                # 进行视角变换，将目标框转换为垂直正视视角
                M = cv2.getPerspectiveTransform(approx, pts_res)
                N = cv2.warpPerspective(img, M, (28, 28), cv2.INTER_NEAREST)
                # 目标框的图像转换为灰度图
                N_gray = cv2.cvtColor(N, cv2.COLOR_BGR2GRAY).astype(np.float32)
                # 像素值归一化到0-1之间
                N_gray /= 255
                # 将图像转换为黑底白字，与mnist数据集中的样本相同
                N_gray = 1 - N_gray
                # 显示图像，为了测试使用
                cv2.imshow("N", N_gray)
                # 将图像转换为网络输入所需形状(1x1x28x28)
                N_gray = np.expand_dims(np.expand_dims(N_gray, axis=0), axis=0)
                # 检测代码不需要自动梯度计算
                with torch.no_grad():
                    N_in = torch.from_numpy(N_gray)
                    N_in = N_in.to(device)
                    outputs = net(N_in)
                # 网络的输出通过.cpu().numpy()转换为numpy格式，可以进行正常操作
                num = np.argmax(outputs.cpu().numpy())
                # 已经获得正确的数字检测结果
                # 在这里你可以加入任何任务级的代码 -----------------

                # 把检测结果在图像左上角显示出来
                cv2.putText(img, 'num: %d' % num, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                pass

    print((cx, cy))
    return img, (cx, cy)


if __name__ == '__main__':
    # 打开摄像头0
    cap = cv2.VideoCapture(0)
    # 加载网络参数，为了手写数字识别
    net = load_mnist_model()

    rospy.init_node('vision', anonymous=True)
    pub = rospy.Publisher('/vision/position', Pose, queue_size=10)

    while True:
        # 读摄像头一帧
        state, frame = cap.read()
        # 提取图像中的方框，并进行手写数字识别
        frame, cxcy = box_extractor(frame, net)
        # 显示图像

        pose = Pose(Point(cxcy[0], cxcy[1], 0), Quaternion(0., 0., 0., 0.))
        pub.publish(pose)
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放内存
    cap.release()
    cv2.destroyAllWindows()
