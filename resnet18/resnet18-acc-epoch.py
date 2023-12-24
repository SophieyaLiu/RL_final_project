import matplotlib.pyplot as plt

# 提供的准确性数据
epochs = list(range(1, 101))
test_accuracy = [
    73.077, 73.077, 73.077, 73.077, 73.077, 73.077, 73.077, 74.359, 78.205, 80.128,
    81.410, 82.051, 82.692, 83.333, 85.256, 83.333, 82.692, 81.410, 82.692, 82.692,
    83.333, 83.974, 84.615, 83.974, 83.974, 83.974, 83.974, 83.974, 83.974, 83.794,
    83.333, 82.692, 82.692, 83.333, 83.974, 83.974, 83.974, 83.333, 83.333, 83.974,
    83.974, 83.974, 84.615, 83.974, 83.974, 84.615, 83.333, 83.974, 83.974, 83.974,
    83.974, 83.974, 83.974, 83.974, 83.974, 83.974, 83.333, 83.333, 83.333, 83.974,
    83.974, 83.974, 83.974, 83.333, 83.974, 84.615, 84.615, 83.974, 83.974, 83.974,
    83.974, 83.333, 83.974, 83.974, 83.974, 84.615, 83.974, 83.974, 83.974, 83.333,
    83.974, 83.974, 83.974, 83.974, 84.615, 83.974, 83.974, 83.974, 83.974, 83.974,
    83.974, 83.974, 83.974, 83.974, 83.974, 83.974, 83.974, 83.974, 83.974, 84.615
]


# 绘制准确性与 epoch 的图像
plt.plot(epochs, test_accuracy, label='Test Accuracy')
plt.title('Test Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# 计算数组的平均值
average_accuracy = sum(test_accuracy) / len(test_accuracy)

# 打印结果
print(f'The average accuracy is: {average_accuracy:.2f}%')