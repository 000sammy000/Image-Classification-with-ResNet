import os
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import uuid

def get_class_counts(base_dir):
    class_counts = {}
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    return class_counts

def augment_image(image_path, save_dir):
    img = Image.open(image_path).convert("RGB")
    base_name, ext = os.path.splitext(os.path.basename(image_path))

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img = img.rotate(angle)

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    brightness_factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    contrast_factor = random.uniform(0.7, 1.5)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    if random.random() < 0.5:
        blur_radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    if random.random() < 0.3:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        cut_size = random.randint(int(0.1 * w), int(0.3 * w))
        x1, y1 = random.randint(0, w - cut_size), random.randint(0, h - cut_size)
        x2, y2 = x1 + cut_size, y1 + cut_size
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

    augmented_img_path = os.path.join(save_dir, f"{base_name}_aug_{uuid.uuid4().hex[:8]}{ext}")
    img.save(augmented_img_path)

    return augmented_img_path

def oversample_classes(base_dir):
    class_counts = get_class_counts(base_dir)
    max_count = 450

    for class_name, count in class_counts.items():
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
            current_count = len(images)
            oversample_count = max_count - current_count
            new_images = []
            while len(new_images) < oversample_count:
                if len(images) == 0:
                    print(f"Label {class_name} has no images to augment, skipping augmentation")
                    break
                image_to_augment = random.choice(images)
                augmented_image = augment_image(image_to_augment, class_path)
                new_images.append(augmented_image)

            print(f"Label {class_name}: Original {count} images, {len(new_images)} augmented images, final {count + len(new_images)} images")

base_dir = 'data/train'
oversample_classes(base_dir)
