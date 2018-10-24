# author    TuringEmmy
# time      2018/10/24 21:00
# project   MachineLearning


from sklearn.preprocessing import StandardScaler

s = StandardScaler()

result = s.fit_transform(
    [
        [2, 2, 10],
        [4, 125, 6]
    ]
)

print(result)

# 在进行实例化
ss = StandardScaler()

result2 = ss.fit(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)
print(ss)

# StandardScaler(copy=True,with_mean=True,with_std=True)

result3 = ss.transform([
    [1, 2, 3],
    [4, 4, 6]
])
print(result3)

# 总结：fit_transform == fit_transform
