import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from mark import BASE_DIR
import numpy as np
import numpy.typing as npt
from nptyping import NDArray, Int, Shape, Float32

# 加载预训练的VGG16模型
vgg16 = models.vgg16(pretrained=True)

# 设定VGG16模型为评估模式
vgg16.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 加载示例图像并进行特征提取
image_path = BASE_DIR/'std.jpg'  # 请替换成您的图像文件路径
image = Image.open(image_path)
image = preprocess(image)
image = image.unsqueeze(0)  # 添加批次维度

# 使用VGG16提取特征
with torch.no_grad():
    features = vgg16.features(image)

avg_pool = nn.AdaptiveAvgPool2d(1)
features = avg_pool(features)
feature_vector = features.view(features.size(0), -1)


feature_vector:  NDArray[Shape["1, 512"], Float32]

print("特征向量的维度:", feature_vector.shape)
v1 = feature_vector[0]
print(feature_vector[0].dtype)

