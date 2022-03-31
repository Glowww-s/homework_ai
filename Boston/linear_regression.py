import numpy as np


def f(x_train, w):  # 线性关系
    x_train = np.insert(x_train, 0, 1, axis=1)
    return np.dot(x_train, w.T)


def loss_fn(y_predict, y_train):  # 损失函数
    return np.sum((y_predict - y_train) ** 2) / len(y_train)  # 均方误差


class LinearRegression:
    def __init__(self, x_train, y_train):
        n = len(x_train[0])  # n元
        self.x_train = x_train
        self.y_train = y_train  # 目标向量
        self.w = np.random.rand(n + 1)  # 权重向量

    def least_square(self):  # 最小二乘法
        x_train = np.insert(self.x_train, 0, 1, axis=1)  # 特征向量补上一项1
        x = np.dot(x_train.T, x_train)
        x = np.linalg.inv(x)  # 求逆
        self.w = np.dot(np.dot(x, x_train.T), self.y_train)

        y_predict = f(self.x_train, self.w)
        loss = loss_fn(y_predict, self.y_train)
        print("最终损失为：{:.8f}".format(loss))  # 输出最终损失
        return self.w

    def grad_descent(self, lr):  # 梯度下降法
        x_train = np.insert(self.x_train, 0, 1, axis=1)  # 特征向量补上一项1
        epoch = 100
        for i in range(epoch):
            dw = np.dot(x_train, self.w) - self.y_train
            dw = 2 * (np.dot(x_train.T, dw))  # 求偏导
            self.w = self.w - lr * dw
            if (i + 1) % 5 == 0:
                y_predict = f(self.x_train, self.w)
                loss = loss_fn(y_predict, self.y_train)
                print("第{}次训练均方误差损失为{:.8f}".format(i + 1, loss))
        return self.w


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    test_x = np.random.normal(0, 1, (506, 3))
    test_y = np.random.normal(0, 1, 506)
    model_1 = LinearRegression(test_x, test_y)
    w_test_1 = model_1.least_square()
    model_2 = LinearRegression(test_x, test_y)
    w_test_2 = model_2.grad_descent(0.001)
    print(w_test_1.shape)
    print(w_test_2.shape)
    print("测试通过")
