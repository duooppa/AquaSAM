import matplotlib.pyplot as plt
import os

# os.path.abspath("../../Desktop")
# os.listdir("../../Desktop")
folder = "./data/Npz_files/output/trainpyt"

import numpy as np
import matplotlib.pyplot as plt

# 加载 npz 文件
# data = np.load(f'{folder}/CT_Abd-Gallbladder_FLARE22_Tr_0001.npz')
# data = np.load(f'{folder}/Tr_000129219.npz')
# data = np.load(f'{folder}/MR_Liver_0000.npz')
# data = np.load(f'{folder}/MR-T2_Brain_Ventricle_0001.npz')
data = np.load(f'{folder}/d_r_122_.npz')


# data = np.load(f'{folder}')

# 显示 npz 文件中的所有文件名
print(data.files)

# 假设你的 .npz 文件包含 'array1'，'array2'，'array3' 三个文件
array_names = data.files
try:
    array1 = data[array_names[0]][0]
    array2 = data[array_names[1]][0]
    array3 = data[array_names[2]][0]
    print("3 arrays, Shape: ", array1.shape,
          array2.shape, array3.shape)
except: # If have two channels instead of three
    array1 = data[array_names[0]]
    array2 = data[array_names[1]]
    print("2 arrays, Shape: ", array1.shape)

# 你可以使用 matplotlib 分别查看这三个数组（即三个灰度图像）

plt.figure(figsize=(10, 4))

# 显示第一个图像
try:
    try:
        plt.subplot(1, 3, 1)
        plt.imshow(array1)
        plt.title('Array 1')

        # 显示第二个图像
        plt.subplot(1, 3, 2)
        plt.imshow(array2)
        plt.title('Array 2')

        # 显示第三个图像
        plt.subplot(1, 3, 3)
        plt.imshow(array3)
        plt.title('Array 3')
    except:
        try:
            plt.subplot(1, 3, 1)
            plt.imshow(array1)
            plt.title('Array 1')

            # 显示第二个图像
            plt.subplot(1, 3, 2)
            plt.imshow(array2)
            plt.title('Array 2')

            # 显示第三个图像
            plt.subplot(1, 3, 3)
            plt.imshow(array3[128])
            plt.title('Array 3')
        except:
            plt.imshow(array1)
            plt.title('Array 1')

except:
    plt.subplot(1, 2, 1)
    plt.imshow(array1[0], cmap='gray')
    plt.title('Array 1')

    # 显示第二个图像
    plt.subplot(1, 2, 2)
    plt.imshow(array2[0], cmap='gray')
    plt.title('Array 2')
# 显示所有图像
plt.tight_layout()
plt.show()
