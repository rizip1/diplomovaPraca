import matplotlib.pyplot as plt

'''
Helper script for creating graphs from hardcoded data
'''

fontsize = 12
fontsizeTitle = 14

x = [60, 90, 120, 150, 180, 210, 240]

y_12_mae = [0.9535, 0.9544, 0.9644, 0.9682, 0.9761, 0.9791, 0.9816]
y_24_mae = [0.9406, 0.9329, 0.9296, 0.9301, 0.9308, 0.9303, 0.9273]

y_12_mse = [1.5891, 1.5825, 1.6075, 1.6121, 1.6273, 1.6404, 1.6458]
y_24_mse = [1.5593, 1.5262, 1.5189, 1.5128, 1.5136, 1.5132, 1.5032]

y_12_bias = [0.0247, 0.0144, 0.0112, 0.0061, 0.0004, 0.0001, -0.0053]
y_24_bias = [0.0229, 0.0048, -0.0056, -0.0091, -0.0112, -0.0078, -0.0100]

plt.figure(figsize=(12, 6))
plt.plot(x, y_12_mae, 'g', label='12 interval')
plt.plot(x, y_24_mae, 'b', label='24 interval')
plt.xticks(x)
plt.legend(loc=2)
plt.title('MAE comparison', fontsize=fontsizeTitle)
plt.ylabel('MAE', fontsize=fontsize)
plt.xlabel('Window length', fontsize=fontsize)
plt.savefig('other/12_to_24_mae.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(x, y_12_mse, 'g', label='12 interval')
plt.plot(x, y_24_mse, 'b', label='24 interval')
plt.xticks(x)
plt.legend(loc=2)
plt.title('MSE comparison', fontsize=fontsizeTitle)
plt.ylabel('MSE', fontsize=fontsize)
plt.xlabel('Window length', fontsize=fontsize)
plt.savefig('other/12_to_24_mse.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(x, y_12_bias, 'g', label='12 interval')
plt.plot(x, y_24_bias, 'b', label='24 interval')
plt.xticks(x)
plt.legend(loc=2)
plt.title('Bias comparison', fontsize=fontsizeTitle)
plt.ylabel('Bias', fontsize=fontsize)
plt.xlabel('Window length', fontsize=fontsize)
plt.savefig('other/12_to_24_bias.png')
plt.close()
