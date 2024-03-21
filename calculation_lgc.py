import os

import imageio
import numpy as np
from pathlib import Path

import utils
from metrics import ssim
from metrics import sam
from metrics import psnr
from metrics import ergas
from metrics import compare_corr
from metrics import RMSE

if __name__ == '__main__':
    dir_TRUE = '/home/clhu/STF/srntt_AdaIN/LGC/calculation/'
    dir_PREDICT = '/home/clhu/STF/srntt_AdaIN/without_tim_AdaIN_ResNet_LGC/test/'
    dir_TRUE = Path(dir_TRUE)
    dir_PREDICT = Path(dir_PREDICT)

    test_dir = dir_PREDICT / 'mertics.csv'  # 存放数据的目录
    test_dir = Path(test_dir)

    psnr_sum = 0
    sam_sum = 0
    ssim_sum = 0
    ergas_sum = 0
    cc_sum = 0
    rmse_sum = 0

    log = utils.get_logger("test")

    for root, dirs, files in os.walk(dir_TRUE):
        for i in range(0, len(files)):
            strname = str(files[i])
            if len(strname)>10:

                img1 = dir_TRUE / strname  ###真实的标签
                img2 = dir_PREDICT / strname
                scale_factor = 0.0001
                # scale_factor = 1
                image1 = imageio.imread(img1)
                im1 = image1.astype(np.float32)  # H*W*C (numpy.ndarray)
                image2 = imageio.imread(img2)
                im2 = image2.astype(np.float32)  # H*W*C (numpy.ndarray)
                # im2 = np.ascontiguousarray(im2.transpose((1,2,0)))
                metadata = {
                    'driver': 'GTiff',
                    'width': 3072,
                    'height': 2560,
                    'count': 4,
                    'dtype': np.int16
                }
                # metadata = {
                #     'driver': 'GTiff',
                #     'width': 1280,
                #     'height': 1792,
                #     'count': 4,
                #     'dtype': np.int16
                # }

                im1 = im1[:, :, 0:4]
                im2 = im2[:, :, 0:4]
                imm1 = im1 * scale_factor
                imm2 = im2 * scale_factor

                psnr_ = psnr(im1, im2)
                sam_ = sam(imm1, imm2)
                ssim_ = ssim(im1, im2)
                ergas_ = ergas(imm2, imm1)
                cc_ = compare_corr(imm1, imm2)
                rmse_ = RMSE(imm1, imm2)
                psnr_sum = psnr_sum + psnr_
                sam_sum = sam_sum + sam_
                ssim_sum = ssim_sum + ssim_
                ergas_sum = ergas_sum + ergas_
                cc_sum = cc_sum + cc_
                rmse_sum = rmse_sum + rmse_
                log.info(f'predict-name[{img2}] - '
                         f'PSNR: {psnr_:.10f} - '
                         f'SAM: {sam_:.10f} - '
                         f'SSIM: {ssim_:.10f} - '
                         f'ERGAS: {ergas_:.10f} - '
                         f'CC: {cc_:.10f} - '
                         f'RSME: {rmse_:.10f}')
                csv_header = ['predict-name', 'PSNR', 'SAM', 'SSIM', 'ERGAS', 'CC', 'RMSE']
                csv_values = [img2, psnr_, sam_, ssim_, ergas_, cc_, rmse_]
                utils.log_csv(test_dir, csv_values, header=csv_header)

        log.info(f'predict-name[average] - '
                 f'PSNR_average: {psnr_sum / 4:.10f} - '
                 f'SAM_average: {sam_sum / 4:.10f} - '
                 f'SSIM_average: {ssim_sum / 4:.10f} - '
                 f'ERGAS_average: {ergas_sum / 4:.10f} - '
                 f'CC_average: {cc_sum / 4:.10f} - '
                 f'RSME_average: {rmse_sum / 4:.10f}')
        csv_header = ['predict-name', 'PSNR', 'SAM', 'SSIM', 'ERGAS', 'CC', 'RMSE']
        csv_values = ['average', psnr_sum / 4, sam_sum / 4, ssim_sum / 4, ergas_sum / 4, cc_sum / 4, rmse_sum / 4]
        utils.log_csv(test_dir, csv_values, header=csv_header)

