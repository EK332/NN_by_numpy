import sys, os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

import matplotlib.pyplot as plt
import numpy as np


class nn_model(object):
    def __init__(self, input_size, class_num, channel=3, batch_size=1, epochs=10, weight_init_std=0.01, lr=0.1,
                 dict=None):
        self.input_size = input_size
        self.class_num = class_num
        self.batch_size = batch_size
        self.channel = channel
        self.epochs = epochs
        self.weight_init_std = weight_init_std
        self.lr = lr
        self.out_dict = {}
        if dict == None:
            self.dict = self.init_dict()
        else:
            self.dict = dict

    def init_dict(self):
        model_dict = {}
        hidden_size = 50

        # w1 = self.weight_init_std * np.random.random((self.input_size, hidden_size))  # (784,50)
        # w1 = self.weight_init_std * np.random.randn(self.input_size, hidden_size) * math.sqrt(1 / 50)  # (784,50)
        w1 = self.weight_init_std * np.random.randn(self.input_size, hidden_size)  # (784,50)
        # b1 = np.random.random((1, w1.shape[1]))
        b1 = np.zeros((1, hidden_size))

        # w2 = self.weight_init_std * np.random.random((w1.shape[1], self.class_num))  # (50,10)
        w2 = self.weight_init_std * np.random.randn(w1.shape[1], self.class_num)  # (50,10)
        # b2 = np.random.random((1, self.class_num))
        b2 = np.zeros((1, self.class_num))

        model_dict['fc1'] = {'w': w1, 'b': b1}
        model_dict['fc2'] = {'w': w2, 'b': b2}
        return model_dict

    def forward(self, x):
        x = self.fc(x, 'fc1')
        x = self.sigmod(x)
        x = self.fc(x, 'fc2')
        x = self.softmax(x)

        return x

    def backward(self, x, labels):
        # forward
        out_fc1 = self.fc(x, 'fc1')
        out_fc1_acti = self.sigmod(out_fc1)
        out_fc2 = self.fc(out_fc1_acti, 'fc2')
        y = self.softmax(out_fc2)

        # save out
        self.out_dict['fc1'] = out_fc1
        self.out_dict['fc1_sigmod'] = out_fc1_acti
        self.out_dict['fc2'] = out_fc2
        self.out_dict['softmax'] = y

        # backward
        # softmax_with_loss反向传播
        grad_softmax_mean_batch_loss = self.d_softmax_mean_batch_loss(y, labels)
        # fc2反向传播
        grad_fc2_dx, grad_fc2_dw, grad_fc2_db = self.d_fc(out_fc1_acti, grad_softmax_mean_batch_loss, 'fc2')
        # sigmod反向传播
        grad_fc1_acti = self.d_sigmod(out_fc1_acti, grad_fc2_dx)
        # relu反向传播
        # grad_fc1_acti = self.d_relu(out_fc1,grad_fc2_dx)
        # fc2反向传播
        _, grad_fc1_dw, grad_fc1_db = self.d_fc(x, grad_fc1_acti, 'fc1')

        # save grads
        grads = {'fc1': {}, 'fc2': {}}
        grads['fc1']['w'] = grad_fc1_dw
        grads['fc1']['b'] = grad_fc1_db
        grads['fc2']['w'] = grad_fc2_dw
        grads['fc2']['b'] = grad_fc2_db

        return grads

    def fc(self, x, fc_name):
        w = self.dict[fc_name]['w']
        b = self.dict[fc_name]['b']
        x = np.matmul(x, w) + b

        return x

    def d_fc(self, fc_input, dout, fc_name):
        dx = np.matmul(dout, self.dict[fc_name]['w'].T)
        dw = np.matmul(fc_input.T, dout)
        db = np.sum(dout, axis=0)  # !!!!why 对上游多个数据同维度的梯度求和
        return dx, dw, db

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmod(self, sigmod_out, dout):
        return dout * sigmod_out * (1 - sigmod_out)

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, relu_input, dout):
        mask = relu_input <= 0
        dout[mask] = 0
        return dout

    def softmax(self, x):
        # 防止e的n次幂数值溢出
        max_vals = np.max(x, axis=1)
        out = []
        for i, _ in enumerate(x):
            exp_vals = np.exp(x[i] - max_vals[i])
            out.append(exp_vals / np.sum(exp_vals))
        return np.array(out)

    def predict(self, inputs):
        return self.forward(inputs)

    def process_label(self, labels):
        out_lables = np.zeros((labels.shape[0], self.class_num))
        for index, label in enumerate(labels):
            out_lables[index][label] = 1
        return out_lables

    def cross_entropy_loss(self, y_label, t_label):
        delta = 1e-7
        loss = -np.sum(t_label * np.log(y_label + delta))
        return loss

    def loss(self, y_labels, t_labels, is_predict=False):
        if is_predict:  # 微分求导反向传播使用
            y_labels = self.predict(y_labels)

        loss = np.array([self.cross_entropy_loss(y, t) for y, t in zip(y_labels, t_labels)])
        # 均值loss
        mean_loss = np.sum(loss) / float(y_labels.shape[0])
        return mean_loss

    def d_softmax_mean_batch_loss(self, softmax_mean_batch_loss_out, labels):
        batch_size = softmax_mean_batch_loss_out.shape[0]
        return (softmax_mean_batch_loss_out - labels) / batch_size

    def gradient_update(self, grads):
        for fc_name in self.dict:
            self.dict[fc_name]['w'] -= self.lr * grads[fc_name]['w']
            self.dict[fc_name]['b'] -= self.lr * grads[fc_name]['b']

    def accuracy(self, x, t):
        y = np.argmax(x, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x_train, t_labels):
        backward_mean_loss = lambda: self.loss(x_train, t_labels, is_predict=True)
        grad_dicts = {}
        for fc_name in self.dict:
            grad_dict = {}
            grad_dict['w'] = self.numerical_diff(self.dict[fc_name]['w'], backward_mean_loss)
            grad_dict['b'] = self.numerical_diff(self.dict[fc_name]['b'], backward_mean_loss)
            grad_dicts[fc_name] = grad_dict

        return grad_dicts

    def numerical_diff(self, x, fun):
        grad = np.zeros_like(x)
        h = 1e-4
        for i, _ in enumerate(x):
            for j, _ in enumerate(x[i]):
                # 保存字典原值
                tmp = x[i][j]
                # 计算f(x+h)
                x[i][j] = float(tmp) + h
                x1 = fun()
                # 计算f(x-h)
                x[i][j] = tmp - h
                x2 = fun()
                # 微积分求导
                grad[i][j] = (x1 - x2) / (2 * h)
                # 还原字典值
                x[i][j] = tmp
        return grad

    def check_grads(self, grads1, grads2):
        for fc_name in grads1.keys():
            for k in grads1[fc_name]:
                n = grads1[fc_name][k]
                g = grads2[fc_name][k]
                diff = np.average(np.abs(n - g))
                print(fc_name + "_" + k + "_diff:" + str(diff))
        print('============')

    def calculate_loss_accu(self, trains, trains_label, tests, tests_label):
        # random_nums = np.random.randint(0, trains.shape[0], x_test.shape[0])
        # trains, trains_label = trains[random_nums], trains_label[random_nums]
        # 训练集损失
        y_train_labels = self.predict(trains)
        train_loss = self.loss(y_train_labels, trains_label)
        # 测试集损失
        y_test_labels = self.predict(tests)
        test_loss = self.loss(y_test_labels, tests_label)
        # 训练集准确率
        train_acc = self.accuracy(y_train_labels, trains_label)
        # 测试集准确率
        test_acc = self.accuracy(y_test_labels, tests_label)

        return train_loss, test_loss, train_acc, test_acc

    def show(self, y1, y2, title):
        x = list(range(len(y1)))
        plt.plot(x, y1, linewidth=2, color='#007700', label='train')
        plt.plot(x, y2, linewidth=2, color='#550000', label='test')
        plt.title(title)
        plt.xlabel("epochs")
        plt.ylabel(title)
        plt.grid(True)
        plt.show()

    def train_0(self, trains, labels, x_test, labels_test):
        # 处理数据集格式
        labels = self.process_label(labels)
        labels_test = self.process_label(labels_test)

        train_num = trains.shape[0]
        acc_train_list = []
        acc_test_list = []
        loss_train_list = []
        loss_test_list = []
        for epoch in range(self.epochs):
            iter_per_epoch = max(int(train_num / self.batch_size), 1)
            for i in range(0, iter_per_epoch):
                random_nums = np.random.randint(0, train_num, self.batch_size)
                train_batch, label__batch = trains[random_nums], labels[random_nums]

                # 方式1：计算图求梯度信息
                grad_dicts_graph = self.backward(train_batch, label__batch)
                # 方式2：微分求梯度信息
                # grad_dicts_numerical = self.numerical_gradient(train_batch, label__batch)
                # # 验证梯度是否一致
                # self.check_grads(grad_dicts_numerical, grad_dicts_graph)

                # 更新梯度
                model.gradient_update(grad_dicts_graph)
                # if i==iter_per_epoch-1 and epoch>20:
                #     self.lr=0.0001

            # 计算准确率和损失
            train_loss, test_loss, train_acc, test_acc = self.calculate_loss_accu(trains, labels, x_test, labels_test)
            # print('各层输出：' + str(self.out_dict))

            print('当前epoch(' + str(epoch) + ')训练集损失:' + str(train_loss))
            print('当前epoch(' + str(epoch) + ')测试集损失:' + str(test_loss))
            print('当前epoch(' + str(epoch) + ')训练集准确率:' + str(train_acc))
            print('当前epoch(' + str(epoch) + ')测试集准确率:' + str(test_acc))
            print('===================')

            loss_train_list.append(train_loss)
            loss_test_list.append(test_loss)
            acc_train_list.append(train_acc)
            acc_test_list.append(test_acc)

        self.show(loss_train_list, loss_test_list, 'loss')
        self.show(acc_train_list, acc_test_list, 'accuracy')

    def train(self, trains, labels, x_test, labels_test, total_iter_nums=10000):
        # 处理数据集格式
        labels = self.process_label(labels)
        labels_test = self.process_label(labels_test)

        train_num = trains.shape[0]
        acc_train_list = []
        acc_test_list = []
        loss_train_list = []
        loss_test_list = []
        iter_per_epoch = train_num / self.batch_size
        for iter_num in range(total_iter_nums):
            random_nums = np.random.randint(0, train_num, self.batch_size)
            train_batch, label__batch = trains[random_nums], labels[random_nums]

            # 方式1：计算图求梯度信息
            grad_dicts_graph = self.backward(train_batch, label__batch)
            # 方式2：微分求梯度信息
            # grad_dicts_numerical = self.numerical_gradient(train_batch, label__batch)
            # 基于方式1和方式2来验证我们的反向传播是否正确，如果梯度误差非常小则正确
            # self.check_grads(grad_dicts_numerical, grad_dicts_graph)

            # 更新梯度
            model.gradient_update(grad_dicts_graph)

            # 计算准确率和损失
            train_loss, test_loss, train_acc, test_acc = self.calculate_loss_accu(trains, labels, x_test, labels_test)
            # print('各层输出：' + str(self.out_dict))
            print('curr_batch_iter(' + str(iter_num) + ')train_loss:' + str(train_loss))
            if iter_num % iter_per_epoch == 0:
                epoch = int(iter_num / iter_per_epoch)
                print('===================')
                print('curr_epoch(' + str(epoch) + ')train_loss:' + str(train_loss))
                print('curr_epoch(' + str(epoch) + ')test_loss:' + str(test_loss))
                print('curr_epoch(' + str(epoch) + ')train_accuracy:' + str(train_acc))
                print('curr_epoch(' + str(epoch) + ')test_accuracy:' + str(test_acc))
                print('===================')

                loss_train_list.append(train_loss)
                loss_test_list.append(test_loss)
                acc_train_list.append(train_acc)
                acc_test_list.append(test_acc)

                if epoch >= self.epochs:
                    break

        self.show(loss_train_list, loss_test_list, 'loss')
        self.show(acc_train_list, acc_test_list, 'accuracy')


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    model = nn_model(input_size=784, class_num=10, channel=1, batch_size=100, epochs=20, lr=0.1)

    # train
    # 博主没时间，暂时还未写权重保存功能。不过其他功能已完善，训练时就能看到模型效果
    model.train(x_train, t_train, x_test, t_test)
    # predict
    # out = model.predict(x_test[:3])
    print('end')
