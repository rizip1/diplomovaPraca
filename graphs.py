import matplotlib.pyplot as plt

x_12 = [60, 90, 120, 150, 180, 210, 240]
x_24 = [60, 120, 180, 240]

y_12_mae = [0.9164, 0.9121, 0.9108, 0.9075, 0.9097, 0.9121, 0.9131]
y_24_mae = [0.9401, 0.9062, 0.8858, 0.8787]

y_12_mse = [1.5071, 1.4802, 1.4755, 1.4608, 1.4668, 1.4767, 1.4784]
y_24_mse = [1.6113, 1.4834, 1.4222, 1.4011]

y_12_bias = [-0.0302, -0.0312, -0.0255, -0.0224, -0.0065, -0.0017, 0.0142]
y_24_bias = [-0.0336, -0.0212, -0.0056, 0.0161]

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.plot(x_12, y_12_mae, 'g', label='12 interval')
plt.plot(x_24, y_24_mae, 'b', label='24 interval')
plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
plt.title('MAE comparison (mean, var)')
plt.ylabel('MAE')
plt.xlabel('Window length')
plt.savefig('other/12_to_24_mae.png')
plt.close()

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.plot(x_12, y_12_mse, 'g', label='12 interval')
plt.plot(x_24, y_24_mse, 'b', label='24 interval')
plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
plt.title('MSE comparison (mean, var)')
plt.ylabel('MSE')
plt.xlabel('Window length')
plt.savefig('other/12_to_24_mse.png')
plt.close()

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.plot(x_12, y_12_bias, 'g', label='12 interval')
plt.plot(x_24, y_24_bias, 'b', label='24 interval')
plt.legend(bbox_to_anchor=(1.02, 1.015), loc=2)
plt.title('Bias comparison (mean, var)')
plt.ylabel('Bias')
plt.xlabel('Window length')
plt.savefig('other/12_to_24_bias.png')
plt.close()
