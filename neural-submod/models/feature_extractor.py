from transformers import ViTFeatureExtractor, ViTModel
from torch.utils.data import BatchSampler, SequentialSampler
import torch

class Feature_Extractor:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        if self.model_name == "dino-cls":
            self.feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
            self.model = ViTModel.from_pretrained('facebook/dino-vitb16')
        elif self.model_name == "resent18":
            raise Exception("feature extractor model not defined")
        else:
            raise Exception("feature extractor model not defined")
    
    def get_feature(self, images):
        sampler = BatchSampler(SequentialSampler(range(len(images))), 32, drop_last=False)
        inputs = []
        for indices in sampler:
            if images[0].mode == 'L':
                images_batch = [images[x].convert('RGB') for x in indices]
            else:
                images_batch = [images[x] for x in indices]
            inputs.append(self.feature_extractor(images_batch, return_tensors="pt"))

        
        img_features = []
        for batch_inputs in inputs:
            tmp_feat_dict = {}
            for key in batch_inputs.keys():
                tmp_feat_dict[key] = batch_inputs[key].to(device=self.device)
            with torch.no_grad():
                batch_outputs = self.model(**tmp_feat_dict)
            batch_img_features = batch_outputs.last_hidden_state[:, 0, :].cpu()
            img_features.append(batch_img_features)
            del tmp_feat_dict

        img_features = torch.cat(img_features, dim=0).to(device=self.device)
        return img_features