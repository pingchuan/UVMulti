import cv2
import os

# 设置输入图像路径和输出视频路径
image_folder = "./logs_uvmt/benthic/demo3/demos88/"
output_video = "./logs_uvmt/benthic/demo2/output88.mp4"
fps = 20  # 设定帧率

# 获取所有图像文件，并按文件名中的数字排序
images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
images = sorted(images, key=lambda x: int(''.join(filter(str.isdigit, x))))  # 仅提取文件名中的数字进行排序

# 确保文件夹不为空
if not images:
    raise ValueError("No images found in the specified directory!")

# 读取第一张图像以获取尺寸
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape
output_dir = os.path.dirname(output_video)
os.makedirs(output_dir, exist_ok=True)
# 设置视频编码器（MP4 格式）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 逐帧写入视频
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# 释放资源
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_video}")

