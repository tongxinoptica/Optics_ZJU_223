import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

import pandas as pd

# 读取 .xlsx 文件
dataframe = pd.read_excel('C:/Users/Administrator/Desktop/paper3/Viridis.xlsx')

lut = dataframe.values

# 假设你有一个lut表，我们使用一个简单的例子
# lut = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
# lut = np.array(lut * 255, dtype=np.uint8)

# 创建一个新的颜色地图
cmap = matplotlib.colors.ListedColormap(lut / 255.0)

# 创建颜色条
fig, ax = plt.subplots(figsize=(100, 10),
                       constrained_layout=True)
img = ax.imshow(np.tile(np.arange(256),(1,10)).reshape(10, 256), cmap=cmap)
plt.axis('off')
#fig.colorbar(img, ax=ax, orientation='horizontal', fraction=.1)
plt.savefig('colorbar.png', dpi=300, bbox_inches='tight')
plt.show()

