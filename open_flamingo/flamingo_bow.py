import os
import re
import json
import copy
import torch
import random
from PIL import Image
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
    tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
    cross_attn_every_n_layers=2
)

model.to(device)

# grab model checkpoint from huggingface hub
checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

caption_path = 'blip2.json'
save_path = 'flamingo_bow.json'

bongard_ow_test = 'assets/data/bongard-ow/bongard_ow_test.json'

query_list = []
with open(caption_path, 'r') as f:
    bongard_ow = json.load(f)
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
        """
        Step 1: Load images
        """
        positive_0 = Image.open(query['positive'][0])
        positive_1 = Image.open(query['positive'][1])
        positive_2 = Image.open(query['positive'][2])
        positive_3 = Image.open(query['positive'][3])
        positive_4 = Image.open(query['positive'][4])
        positive_5 = Image.open(query['positive'][5])

        negative_0 = Image.open(query['negative'][0])
        negative_1 = Image.open(query['negative'][1])
        negative_2 = Image.open(query['negative'][2])
        negative_3 = Image.open(query['negative'][3])
        negative_4 = Image.open(query['negative'][4])
        negative_5 = Image.open(query['negative'][5])

        query_image = Image.open(query['query'])

        """
        Step 2: Preprocessing images
        Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
        batch_size x num_media x num_frames x channels x height x width. 
        In this case batch_size = 1, num_media = 3, num_frames = 1,
        channels = 3, height = 224, width = 224.
        """
        vision_x = [image_processor(positive_0).unsqueeze(0),
                    image_processor(positive_1).unsqueeze(0),
                    image_processor(positive_2).unsqueeze(0),
                    image_processor(positive_3).unsqueeze(0),
                    image_processor(positive_4).unsqueeze(0),
                    image_processor(positive_5).unsqueeze(0),
                    image_processor(negative_0).unsqueeze(0),
                    image_processor(negative_1).unsqueeze(0),
                    image_processor(negative_2).unsqueeze(0),
                    image_processor(negative_3).unsqueeze(0),
                    image_processor(negative_4).unsqueeze(0),
                    image_processor(negative_5).unsqueeze(0),
                    image_processor(query_image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)


        """
        Step 3: Preprocessing text
        Details: In the text we expect an <image> special token to indicate where an image is.
        We also expect an <|endofchunk|> special token to indicate the end of the text 
        portion associated with an image.
        """
        tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        lang_x = tokenizer(["Given 6 'positive' images and 6 'negative' images, where 'positive' images share 'common' visual concepts and 'negative' images cannot, the 'common' visual concepts exclusively depicted by the 'positive' images. And then given 1 'query' image, please determine whether it belongs to 'positive' or 'negative'.\n'positive' images:<|endofchunk|><image><image><image><image><image><image>\n'negative' images:<|endofchunk|><image><image><image><image><image><image>\n'query' image:<|endofchunk|><image>\n'query' image belongs to"],
            return_tensors="pt",
        ).to(device)


        """
        Step 4: Generate text
        """
        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
        )
        generated_text = tokenizer.decode(generated_text[0])

        answer = re.findall("'(.*?)'", generated_text.split("'query'")[-1])
        query['answer'] = answer[0].lower()
        summary.append(copy.deepcopy(query))
        
    with open(save_path, "w") as file:
        json.dump(summary, file, indent=4)