import copy
import json

concept_path = 'instructBLIP_gpt4_concepts_query.json'
save_path = 'instructBLIP_gpt4_concepts_reasoning.json'

def main():
    with open(concept_path, 'r') as f:
        bongard_ow = json.load(f)

        summary = []
        for sample in bongard_ow:
            uid = sample['uid']

            del sample['imageFiles']
            del sample['captions']
            concepts = sample['concepts']
            for i in range(14):
                concepts[i] = {cpt.strip() for cpt in concepts[i].split(',')}
            
            positive_intersection = concepts[0] & concepts[1] & concepts[2] & \
                                    concepts[3] & concepts[4] & concepts[5]
            sample['intersection'] = ','.join(positive_intersection)

            query = concepts[6]
            sample['uid'] = uid + '_A'
            sample['query'] = ','.join(list(query))
            if positive_intersection.issubset(query):
                sample["answer"] = 'positive'
            else:
                sample["answer"] = 'negative'
            summary.append(copy.deepcopy(sample))

            query = concepts[13]
            sample['uid'] = uid + '_B'
            sample['query'] = ','.join(list(query))
            if positive_intersection.issubset(query):
                sample["answer"] = 'positive'
            else:
                sample["answer"] = 'negative'
            summary.append(copy.deepcopy(sample))
        
        for s in summary:
            for i in range(14):
                s['concepts'][i] = ','.join(list(s['concepts'][i]))
        
        with open(save_path, "w") as file:
            json.dump(summary, file, indent=4)

if __name__ == '__main__':
    main()