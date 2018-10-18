# author    TuringEmmy
# time      2018/10/18 22:02
# project   MachineLearning


from sklearn.preprocessing import StandardScaler


#-----------------------------------标准化缩放-----------------------------------
def stand():
    """标准化缩放"""
    # 实例化
    std = StandardScaler()
    source = [[1., -1., 3.],
              [2., 4., 2.],
              [4., 6., -1.]]
    data = std.fit_transform(source)
    print(data)


if __name__ == '__main__':
    stand()
