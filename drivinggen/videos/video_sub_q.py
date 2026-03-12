import pyiqa
import pyiqa.models
import pyiqa.models.inference_model
import torch
from PIL import Image
from pyiqa.models.inference_model import InferenceModel
from torchvision import transforms
from torchvision.transforms import ToTensor
from typing import Tuple
import numpy as np
from .metrics.base_metrics import open_image
from tqdm import tqdm

subjective_quality_model = None


def init_subjective_quality_model():
    global subjective_quality_model
    if subjective_quality_model is not None:
        return
    
    subjective_quality_model = pyiqa.create_metric(
        'clipiqa+',
    )


def subjective_quality_zscore_rescale_infer(arr, target_low=0.0, target_high=1.0, gt=None):
    """把 1-D 数组 z-score 后再线性映射到 [target_low, target_high]"""
    arr = np.asarray(arr, dtype=np.float32)
    gt_mean, gt_std, gt_z_min, gt_z_max = gt
    score_type = True
    z = (arr - gt_mean) / (gt_std + 1e-9)

    scaled = np.clip((z - gt_z_min) / (gt_z_max - gt_z_min + 1e-9), 0, 1)

    if score_type == True:
        score = scaled
    elif score_type == False:
        score = 1 - scaled
    elif score_type == 'mid':
        score = 1 - 2 * (scaled - 0.5) ** 2
    elif score_type == 'row':
        # ① “隐噪区”——scaled < 0.2 直接记满分 1
        # ② 否则按单向低好：1 - ((x-0.2)/0.8)，最后 clip 到 [0,1]
        score = np.where(
            scaled < 0.2,
            1.0,
            1.0 - np.clip((scaled - 0.2) / 0.8, 0.0, 1.0)
        )
    return score * (target_high - target_low) + target_low

@torch.no_grad()
def get_subjective_quality(video_list):
    scores = []

    global subjective_quality_model
    init_subjective_quality_model()
    preprocessing = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            # transforms.ToTensor(),
        ]
    )

    def _process_image(
        rendered_images,
    ) -> float:
        preprocessing = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        rendered_images_: List[torch.Tensor] = []
        for image in rendered_images:
            # Handle the rendered image input
            if isinstance(image, str):
                image = preprocessing(open_image(image))
            else:
                image = preprocessing(image)
            rendered_images_.append(image)

        img: torch.Tensor = torch.stack(rendered_images_).to('cuda')

        return img
    print(f'=========================Start Subjective Quality=========================')
    for rendered_images in tqdm(video_list):
        imgs = _process_image(rendered_images)

        with torch.no_grad():
            score = subjective_quality_model(imgs)
        score = score.mean()
        scores.append(score.item())

    print(f'raw: {np.array(scores).mean()}') # gt 0.554

    return float(np.array(scores).mean())
    return float(np.array(scores).mean()), np.array(scores)

    # if not os.path.exists('./subjective_quality_5_95.pkl'):
    #     normed_stat_gt = {}
    #     mu, sigma, z_min, z_max = zscore_rescale_gt(scores, q_low=5, q_high=95)
    #     normed_stat_gt['clipiqa+'] = [mu, sigma, z_min, z_max]
    #     with open('./subjective_quality_5_95.pkl', 'wb') as f:
    #         pickle.dump(normed_stat_gt, f)
    # else:
    #     with open('./subjective_quality_5_95.pkl', 'rb') as f:
    #         subjective_quality_gt = pickle.load(f)
    
    score = subjective_quality_zscore_rescale_infer(scores, gt=subjective_quality_gt['clipiqa+'])

    print(np.array(score).mean())   # gt 0.51

    return np.array(score).mean()