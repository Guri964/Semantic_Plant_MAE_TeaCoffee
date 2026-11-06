import cv2
from torchvision import transforms
from PIL import Image
from config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD


val_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
])


def load_image_bgr(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f'Could not decode image: {image_path}')
    img_bgr = cv2.resize(img_bgr, (256, 256))
    return img_bgr


def prepare_tensor(image_path, device):
    img = Image.open(image_path).convert('RGB')
    tensor = val_transforms(img).unsqueeze(0).to(device)
    return tensor