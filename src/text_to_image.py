from PIL import ImageDraw, ImageFont, Image
import torchvision.transforms as T

image_size = (1024, 1024)
image = Image.new("RGB", image_size, color=(255, 255, 255))
text = """Voici un diagnostic de l'erreur et les étapes pour la résoudre.

Diagnostic de l'erreur ValueError: Unrecognized model
L'erreur que vous rencontrez est levée par la bibliothèque transformers\n(AutoConfig.from_pretrained)\npour une raison très précise : elle ne parvient pas à identifier l'architecture du modèle que vous essayez de charger.
La cause principale est que l'identifiant de modèle junkim100/DeepSeek-3B-MoE-decoder\nn'est pas un dépôt de modèle valide sur le Hub Hugging Face, ou qu'il ne\ns'agit pas d'un chemin local pointant vers un modèle valide."""

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
    # save the image

    image_path = "image.png"
    image_with_text.save(image_path)