import os
import copy
import json
import torch
import numpy as np
from PIL import Image
from lavis.models import model_zoo, load_model_and_preprocess
from collections import Counter

caption_path = 'instructBLIP_gpt4_concepts.json'
save_path = 'instructBLIP_gpt4_concepts_query.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

def main():
    with open(caption_path, 'r') as f:
        bongard_ow = json.load(f)

        for sample in bongard_ow:
            imageFiles = [os.path.join('assets/data/bongard-ow', imageFile) for imageFile in sample['imageFiles']]
            images = [vis_processors["eval"](Image.open(imageFile).convert("RGB")).numpy() for imageFile in imageFiles]
            images = torch.from_numpy(np.array(images)).to(device)

            concepts = sample['concepts']
            for i in range(14):
                concepts[i] = list({cpt.strip() for cpt in concepts[i].split(',')})
                
            positive_union = concepts[0] + concepts[1] + concepts[2] + concepts[3] + \
                            concepts[4] + concepts[5]
            count_positive_union = Counter(positive_union)
            filter_positive_union = set(filter(lambda x: count_positive_union[x] >= 3, count_positive_union))

            negative_union = concepts[7] + concepts[8] + concepts[9] + concepts[10] + \
                            concepts[11] + concepts[12]
            count_negative_union = Counter(negative_union)
            filter_negative_union = set(filter(lambda x: count_negative_union[x] >= 2, count_negative_union))

            concept_union = filter_positive_union | filter_negative_union

            for i in range(14):
                concepts[i] = set(concepts[i])
                concepts_diff = concept_union - concepts[i]

                if len(concepts_diff) == 0:
                    concepts[i] = ','.join(list(concepts[i]))
                    continue

                for diff in concepts_diff:
                    prompt = f'Does the image depict "{diff}"?'
                    response = model.generate({"image": images[i].unsqueeze(0), "prompt": prompt})[0]
                    
                    if len(response) >= 3:
                        flag = response[:3].strip(',').lower()
                        if flag == 'yes':
                            concepts[i].update({diff})

                concepts[i] = ','.join(list(concepts[i]))

            sample['concepts'] = copy.deepcopy(concepts)
            with open(save_path, "w") as file:
                json.dump(bongard_ow, file, indent=4)

if __name__ == '__main__':
    main()