import os
import torch
from transformers import CLIPProcessor, CLIPModel
from openvino.runtime import Core
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Initialize CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
clip_model1 = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Initialize OpenVINO core and load the quantized model
core = Core()
model_path = "/app/models/clip-openvino/clip-vit-base-patch32_int8.xml"
compiled_model = core.compile_model(core.read_model(model_path), "CPU")

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def toto_clip(model, processor, image1, image2, image3, text1, text2, text3):
    inputs = processor(text=[text1, text2, text3], images=[image1, image2, image3], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    x1 = outputs.image_embeds[0]
    x2 = outputs.image_embeds[1]
    x3 = outputs.image_embeds[2]
    sim_x1_x2 = torch.nn.functional.cosine_similarity(x1, x2, dim=0)
    sim_x1_x3 = torch.nn.functional.cosine_similarity(x1, x3, dim=0)
    return sim_x1_x2 > sim_x1_x3

def toto_openvino(model, processor, image1, image2, image3, text1, text2, text3):
    image_inputs = processor(images=[image1, image2, image3], return_tensors="np", padding=True)
    infer_request = model.create_infer_request()
    infer_request.infer({'pixel_values': image_inputs['pixel_values']})
    image_features = infer_request.get_tensor(model.output(0)).data

    text_inputs = processor(text=[text1, text2, text3], return_tensors="pt", padding=True)
    text_outputs = clip_model1.get_text_features(**text_inputs)

    x1 = image_features[0]
    x2 = image_features[1]
    x3 = image_features[2]

    if x1.size == 0 or x2.size == 0 or x3.size == 0:
        return False

    sim_x1_x2 = cosine_similarity(x1.reshape(1, -1), x2.reshape(1, -1))
    sim_x1_x3 = cosine_similarity(x1.reshape(1, -1), x3.reshape(1, -1))
    return sim_x1_x2 > sim_x1_x3

# Example usage
if __name__ == "__main__":
    # Example image and text paths
    image1_path = "path/to/image1.jpg"
    image2_path = "path/to/image2.jpg"
    image3_path = "path/to/image3.jpg"
    text1 = "Example text 1"
    text2 = "Example text 2"
    text3 = "Example text 3"

    image1 = load_image(image1_path)
    image2 = load_image(image2_path)
    image3 = load_image(image3_path)

    result1 = toto_clip(clip_model1, clip_processor, image1, image2, image3, text1, text2, text3)
    result2 = toto_openvino(compiled_model, clip_processor, image1, image2, image3, text1, text2, text3)

    print(f"CLIP model result: {result1}")
    print(f"OpenVINO model result: {result2}")
