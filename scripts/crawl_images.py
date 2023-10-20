import os
import json
import urllib.request

def main():
    bongard_ow = json.load(open('assets/data/bongard-ow/bongard_ow.json', 'r'))

    for sample in bongard_ow:
        uid = sample['uid']
        save_dir = os.path.join('assets/data/bongard-ow/images', uid)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        imageFiles = sample['imageFiles']

        urls = sample['urls']
        for i in range(len(urls)):
            url = urls[i]
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'})
                urllib.request.urlopen(req)
                save_path = os.path.join('assets/data/bongard-ow', imageFiles[i])
                with open(save_path, "wb") as f:
                    with urllib.request.urlopen(req) as r:
                        f.write(r.read())
            except Exception as e:
                print(f'download eorror: {e}')
                print(f'image save path: {imageFiles[i]}')
                print(f'url: {url}\n')

if __name__ == '__main__':
    main()