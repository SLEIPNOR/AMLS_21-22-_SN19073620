# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 06:29:42 2021

@author: Blade
"""

# -*- coding: UTF-8 -*-
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_random_data(num):
    random_data = list()
    for i in range(0, num):
        # random_data.append(random.random())
        random_data.append(random.randint(0, 100))
    return random_data


def csdn_plot_3D_scatter():
    """
        plot the 3D scatter picture.
    """
    '''get data ready.'''
    number_of_point = 100
    x_data = range(0, number_of_point)
    y_data_1 = get_random_data(number_of_point)
    y_data_2 = get_random_data(number_of_point)
    y_data_3 = get_random_data(number_of_point)

    '''plot starting ... '''
    fig = plt.figure()
    plt.rcParams['savefig.dpi'] = 1000       # 图片像素
    plt.rcParams['figure.dpi'] = 1000        # 分辨率
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=x_data, ys=0, zs=y_data_1, c='#4d3333', s=12, alpha=1, label='English', marker='*')
    ax.scatter(xs=x_data, ys=1, zs=y_data_2, c='#3333cc', s=12, alpha=1, label='Chinese', marker='o')
    ax.scatter(xs=x_data, ys=2, zs=y_data_3, c='#ff1493', s=12, alpha=1, label='Math', marker='^')
    ax.set_xticklabels([" ", " ", "Students", " ", " "], fontsize=20)
    ax.set_yticklabels(["English", "Chinese", "Math"], fontsize=20)
    ax.set_zlabel('Score', fontsize=16)
    ax.set_xticks([0, 20, 40, 60, 80, 100])	 # x 轴刻度密度
    ax.set_yticks([0, 1, 2])			     # y 轴刻度密度
    ax.set_xlim(left=0, right=100)           # x 轴显示范围
    ax.set_ylim(bottom=0, top=2)             # y 轴显示范围
    plt.tick_params(labelsize=13)		     # 刻度字体大小
    plt.tight_layout(rect=(0, 0, 1, 1))
    # plt.savefig('student_score.pdf')
    plt.show()


if __name__ == '__main__':

    csdn_plot_3D_scatter()