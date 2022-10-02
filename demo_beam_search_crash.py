import requests
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder

def generate_caption(image, device):
    gpu_image = transforms.Compose([
        transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])(image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=20, min_length=5)
    return caption[0]


blip_image_eval_size = 384
blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
blip_model = blip_decoder(pretrained=blip_model_url, image_size=blip_image_eval_size, vit='base', med_config = 'BLIP/configs/med_config.json')
blip_model.eval()

image_path_or_url = "https://cdnb.artstation.com/p/assets/images/images/032/142/769/large/ignacio-bazan-lazcano-book-4-final.jpg" #@param {type:"string"}
if str(image_path_or_url).startswith('http://') or str(image_path_or_url).startswith('https://'):
    image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert('RGB')
else:
    image = Image.open(image_path_or_url).convert('RGB')



print("running on CPU")
blip_model = blip_model.to("cpu")
caption = generate_caption(image, "cpu")
print(" -> got caption", caption)

print("running on MPS")
blip_model = blip_model.to("mps")
caption = generate_caption(image, "mps")
print(" -> got caption", caption)
