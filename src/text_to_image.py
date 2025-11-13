from PIL import ImageDraw, ImageFont, Image
import torchvision.transforms as T

image_size = (1024, 1024)
image = Image.new("RGB", image_size, color=(255, 255, 255))
text = "A scenic view of mountains during sunrise"

def add_text_to_image(image: Image.Image, text: str) -> Image.Image:

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except IOError:
        font = ImageFont.load_default()

    text_position = (10, 10)
    text_color = (0, 0, 0)  # Black text
    draw.text(text_position, text, fill=text_color, font=font)
    return image

if __name__ == "__main__":
    image_with_text = add_text_to_image(image, text)
    image_with_text.show()

    # convert to tensor
    transform = T.ToTensor()
    image_tensor = transform(image_with_text)
    print("Image tensor shape:", image_tensor.shape)