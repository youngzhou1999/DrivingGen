import sys
import os
# sys.path.append('/mnt/cache/zhouyang/dg-bench/common_metrics_on_video_quality')
# from calculate_fvd import calculate_fvd

sys.path.append(os.path.abspath('third_parties/stylegan-v'))
from src.scripts.calc_metrics_for_dataset import calc_metrics_

# def get_fvd(video_tensor_1, video_tensor_2,):
#     fvd = calculate_fvd(video_tensor_1, video_tensor_2, 'cuda', method='styleganv', only_final=True)
#     return fvd['value'][0]


def get_fvd(fake_path, gt_path):
    # 128f = 100
    print('cal fvd.')
    fvd = calc_metrics_(
        # metrics=['fvd2048_16f', 'fvd2048_128f', 'fvd2048_128f_subsample8f'],
        metrics=['fvd2048_100f'],
        real_data_path=gt_path,
        fake_data_path=fake_path,
        mirror=False,
        resolution=256,
        gpus=1,
        verbose=True,
        use_cache=False,
        num_runs=1
    )
    # return [fvd[0]['results']['fvd2048_16f'], fvd[1]['results']['fvd2048_128f'], fvd[2]['results']['fvd2048_128f_subsample8f']]
    # 100 frames
    return fvd[0]['results']['fvd2048_100f']