import mimetypes
import os
import re
import copy
import json
import random
from io import BytesIO
from typing import Union
import cv2
import torch
import transformers
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
import sys

sys.path.append("../..")

from otter.modeling_otter import OtterForConditionalGeneration


# ------------------- Utility Functions -------------------
def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------
def get_image(url: str) -> Union[Image.Image, list]:
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    else:
        raise ValueError("Invalid content type. Expected image or video.")


# ------------------- OTTER Response Functions -------------------
def get_response(image_list, model=None, image_processor=None) -> str:
    input_data = image_list

    if isinstance(input_data, Image.Image):
        vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    elif isinstance(input_data, list):  # list of video frames
        vision_x = image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")

    lang_x = model.text_tokenizer(
        ["Given 6 'positive' images and 6 'negative' images, where 'positive' images share 'common' visual concepts and 'negative' images cannot, the 'common' visual concepts exclusively depicted by the 'positive' images. And then given 1 'query' image, please determine whether it belongs to 'positive' or 'negative'.\n'positive' images:<|endofchunk|><image><image><image><image><image><image>\n'negative' images:<|endofchunk|><image><image><image><image><image><image>\n'query' image:<|endofchunk|><image>\n'query' image belongs to"],
        return_tensors="pt",
    )
    # Get the data type from model's parameters
    model_dtype = next(model.parameters()).dtype

    # Convert tensors to the model's data type
    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
    )
    generated_text = tokenizer.decode(generated_text[0])
    print(generated_text)

    parsed_output = re.findall("'(.*?)'", generated_text.split("'query'")[-1])
    return parsed_output


# ------------------- Main Function -------------------
if __name__ == "__main__":
    load_bit = "bf16"
    # dtype = torch.bfloat16 if load_bit == "bf16" else torch.float32
    precision = {}
    if load_bit == "bf16":
        precision["torch_dtype"] = torch.bfloat16
    elif load_bit == "fp16":
        precision["torch_dtype"] = torch.float16
    elif load_bit == "fp32":
        precision["torch_dtype"] = torch.float32
    model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-MPT7B", device_map="sequential", **precision)
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()

    save_path = 'otter_bow.json'
    bongard_ow_test = 'assets/data/bongard-ow/bongard_ow_test.json'

    query_list = []
    with open(bongard_ow_test, 'r') as f:
        bongard_ow = json.load(f)[:]
        for sample in bongard_ow:
            uid = sample['uid']
            
            image_paths = os.listdir(os.path.join('assets/data/bongard-ow/images', uid))
            image_paths.sort()
            image_paths[:7], image_paths[7:] = image_paths[7:], image_paths[:7]
            for i in range(len(image_paths)):
                image_paths[i] = copy.deepcopy(os.path.join('assets/data/bongard-ow/images', uid, image_paths[i]))

            commonSense = sample['commonSense']
            concept = sample['concept']
            caption = sample['caption']

            query = {}
            query['commonSense'] = commonSense
            query['concept'] = concept
            query['caption'] = caption

            query['uid'] = uid + '_A'
            query['positive'] = image_paths[:6]
            query['negative'] = image_paths[7:13]
            query['query'] = image_paths[6]
            query_list.append(copy.deepcopy(query))

            query['uid'] = uid + '_B'
            query['positive'] = image_paths[:6]
            query['negative'] = image_paths[7:13]
            query['query'] = image_paths[13]
            query_list.append(copy.deepcopy(query))
        
        random.shuffle(query_list)

        summary = []
        for query in query_list:
            urls = query['positive'] + query['negative'] + [query['query']]

            encoded_frames_list = []
            for url in urls:
                frames = get_image(url)
                encoded_frames_list.append(frames)

            response = get_response(encoded_frames_list, model, image_processor)
            print(f"response: {response}")
            
            try:
                query['answer'] = response[0].lower()
                print(query['answer'])
            except:
                continue

            summary.append(copy.deepcopy(query))
            
        with open(save_path, "w") as file:
            json.dump(summary, file, indent=4)