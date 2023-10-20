import os
import copy
import json
import torch
import argparse
import numpy as np
from PIL import Image
from lavis.models import model_zoo, load_model_and_preprocess

def main(args):
    vlm = args.vlm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt" if vlm == 'blip2' else "blip2_vicuna_instruct",
        model_type="caption_coco_opt6.7b" if vlm == 'blip2' else "vicuna7b",
        is_eval=True,
        device=device)

    image_path = 'assets/data/bongard-ow/bongard_ow_test.json'
    caption_path = f'{vlm}.json'

    captions = []
    with open(image_path, 'r') as f:
        bongard_ow_test = json.load(f)
        for sample in bongard_ow_test:
            uid = sample['uid']
            imageFiles = [os.path.join('assets/data/bongard-ow', imageFile) for imageFile in sample['imageFiles']]

            # preprocess the image
            # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
            images = [
                vis_processors["eval"](Image.open(imageFile).convert("RGB")).numpy()
                for imageFile in imageFiles
            ]
            images = torch.from_numpy(np.array(images)).to(device)

            # generate caption
            sample['captions'] = [model.generate({"image": images[i].unsqueeze(0), "prompt": "Write a detailed description."})[0] for i in range(14)]
            captions.append(copy.deepcopy(sample))

        with open(caption_path, "w") as file:
            json.dump(captions, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm', type=str, choices=['blip2', 'instructBLIP'], help='choose a caption model')
    
    args = parser.parse_args()
    main(args)