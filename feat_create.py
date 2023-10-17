from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModel
import json
import os
from PIL import Image
import torch


if __name__ == '__main__':
    train_json = '/home/palm/data/coco/annotations/annotations/captions_train2017.json'
    val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
    src_dir = "/home/palm/data/coco/images"
    dst_dir = "/home/palm/data/coco/features"
    feature_extractor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
    for folder, file in zip(['train2017', 'val2017'], [train_json, val_json]):
        data = json.load(open(file))
        for image in data['images']:
            img = Image.open(os.path.join(src_dir, folder, image['file_name']))
            pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values.cuda()
            m = model.vision_model(pixel_values=pixel_values)
            torch.save(
                {'pooler_output': m['pooler_output'], 'last_hidden_state': m['last_hidden_state']},
                os.path.join(dst_dir, image['file_name']+'.pth')
            )
