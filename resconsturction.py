from Models import FCN
import os
import argparse
import openslide
import numpy as np
from keras.utils import multi_gpu_model
import time
import ScanNet2_deploy 

import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure, color, io, filters
import tensorflow as tf
from preprocess.utils import openwholeslide, filter_ch, overlap1D, intersect, get_sample_mask

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)




def merge_bbox(bboxes: list):
    """
    整合bbox, 将有相交的整合成更大的bbox
    :param bboxes:
    :return:
    """
    for idx in range(len(bboxes)):
        current_bbox = bboxes[idx]
        for next_idx in range(idx + 1, len(bboxes)):
            other_bbox = bboxes[next_idx]
            if intersect(current_bbox, other_bbox):
                minr = min(current_bbox[0], other_bbox[0])
                minc = min(current_bbox[1], other_bbox[1])
                maxr = max(current_bbox[2], other_bbox[2])
                maxc = max(current_bbox[3], other_bbox[3])
                bboxes.pop(idx)
                bboxes.pop(next_idx - 1)
                bboxes.append((minr, minc, maxr, maxc))
                return merge_bbox(bboxes)
    return bboxes


def pre(m, id, x, y, n=83, alpha=4):
    ws_path = args.data_path + f'Images/{id}.tif'
    # print(ws_path)

    strides = int(32 / alpha)
    size = (n - 1) * 32 + 244

    reader = openslide.OpenSlide(ws_path)
    img = reader.read_region((y, x), 0, (size + strides * (alpha - 1), size + strides * (alpha - 1))).convert('RGB')
    img = np.asarray(img)
    # print("readed..",time.time()-start)

    # import cv2
    # cv2.imwrite('prediction/150/test.bmp',img)

    img = img.astype('float32')
    img /= 255.
    _input = []
    prediction = np.zeros((n * alpha, n * alpha))
    for iter in range(alpha * alpha):
        i = int(iter / alpha)
        j = int(iter % alpha)
        _input.append(img[i * strides:i * strides + size, j * strides:j * strides + size, :])
    _input = np.asarray(_input)
    assert _input.shape == (alpha * alpha, size, size, 3)
    _prediction = m.predict(_input, batch_size=1)[:, :, :, 1]
    # print("predicted..",time.time()-start)
    for iter in range(alpha * alpha):
        i = int(iter / alpha)
        j = int(iter % alpha)
        for k1 in range(n):
            for k2 in range(n):
                prediction[i + k1 * alpha, j + k2 * alpha] = _prediction[iter, k1, k2]
    # np.save('prediction/150/pre.npy',prediction)
    # cv2.imwrite('prediction/150/test_mask.bmp', prediction*255)
    return prediction


def DPTS(id, n=83, flag='pre', alpha=4):
    if not os.path.exists(f'prediction/{id}/'):
        os.mkdir(f'prediction/{id}')
    size = (n - 1) * 32 + 244
    DPT_strides = 32 * n
    start = time.time()
    m = ScanNet2_deploy.GetModel(flag='pre', size=size)
    if args.multi_gpu>1:
        m = multi_gpu_model(m, gpus=args.multi_gpu)
    m.load_weights(args.load_weights, by_name=True)

    print("loaded..", time.time() - start)
    tif_path = f'/home/data/ACDC/Images/{id}.tif'
    mask_reader = openslide.open_slide(f'/home/data/ACDC/Mask/{id}_mask.tif')
    slide_reader, levels, dims = openwholeslide(tif_path)
    w, h = dims[6]
    scale_factor_row, scale_factor_col = float(
        dims[0][1]) / h, float(dims[0][0]) / w
    thumbnail = np.asarray(slide_reader.get_thumbnail((w, h)).convert('RGB'))
    sample_mask = get_sample_mask(thumbnail)
    label_image = measure.label(sample_mask)
    sample_bbox = []
    for region in measure.regionprops(label_image):
        if region.area > 10:
            sample_bbox.append(region.bbox)
    sample_bbox = merge_bbox(sample_bbox)
    print(scale_factor_row, scale_factor_col)
    for iter, region_bbox in enumerate(sample_bbox):
        print('*'*30)
        print("REGION BBOX:", region_bbox)
        start = time.time()
        xmin = int(region_bbox[0] * scale_factor_row)
        ymin = int(region_bbox[1] * scale_factor_col)
        xmax = int(region_bbox[2] * scale_factor_row)
        ymax = int(region_bbox[3] * scale_factor_col)

        col_count = int((ymax - ymin-int(32 * (alpha - 1) / alpha)-size) / DPT_strides)+2
        row_count = int((xmax - xmin-int(32 * (alpha - 1) / alpha)-size) / DPT_strides)+2
        gt = np.array(mask_reader.read_region((ymin, xmin), 3, (col_count*alpha*n*4, row_count*alpha*n*4)).convert('RGB'))[:,:,0]   
        if np.sum(gt)==0:
            print("No label...")
            continue     
        r = 0
        for r in range(row_count):
            for c in range(col_count):
                print(xmin + r * DPT_strides, ymin + c * DPT_strides)
                if c == 0:
                    _prediction = pre(m, id, r * DPT_strides + xmin, c * DPT_strides + ymin, alpha=alpha)
                else:
                    tmp = pre(m, id, r * DPT_strides + xmin, c * DPT_strides + ymin, alpha=alpha)
                    _prediction = np.hstack((_prediction, tmp))
            if r == 0:
                prediction = _prediction.copy()
            else:
                prediction = np.vstack((prediction, _prediction))
        post = np.zeros((prediction.shape[0]*4, prediction.shape[1]*4))
        scale = 4
        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                post[scale*i:scale*(i+1),scale*j:scale*(j+1)] = prediction[i,j]   
        try:
            #cv2.imwrite(f'prediction/{id}/{xmin}_{ymin}_{xmax}_{ymax}.jpg',
            #           np.array(slide_reader.read_region((ymin, xmin), 3, (col_count*alpha*n, row_count*alpha*n)).convert('RGB'))[:,:,::-1])
            #cv2.imwrite(f'prediction/{id}/{xmin}_{ymin}_{xmax}_{ymax}_mask.jpg', gt*255)
            #np.save(f'prediction/{id}/{xmin}_{ymin}_{xmax}_{ymax}.npy', prediction)
            
            dice = []
            thresh = 0.7
            post = post>thresh
            dice.append(2*np.sum(gt*post)/(np.sum(gt)+np.sum(post)))
            print(dice)
        except Exception as es:
            print(es)
            print('Fail to save the image!')
            pass
        end = time.time()
        import csv
        if not os.path.exists(f'prediction/fcn/'):
            os.mkdir(f'prediction/fcn/')
        with open(f'prediction/fcn/history.csv', 'a+') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([id, xmin, xmax, ymin, ymax])
            spamwriter.writerow(dice)
        np.save(f'prediction/fcn/{id}_{xmin}_{ymin}_{xmax}_{ymax}.npy', post)
        print(end - start)
        print('*'*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_gpu", type=int, default=1)
    parser.add_argument("--load_weights", type=str, default="")
    parser.add_argument("--data_path", type=str, default='/home/data/ACDC/')
    args = parser.parse_args()
    dataset = {'test': [6, 17, 27, 34, 51, 75, 82, 87, 139, 143],
                'train': [1, 2, 3, 4, 5, 9, 10, 12, 13, 14, 16, 18, 19, 20, 22, 24
                , 25, 26, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 
                43, 44, 45, 46, 47, 50, 52, 53, 54, 56, 57, 58, 60, 61, 63, 64, 65, 
                66, 67, 68, 69, 70, 72, 73, 74, 76, 77, 78, 79, 80, 83, 84, 85, 88,
                89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                105, 106, 107, 108, 110, 111, 112, 113, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124, 125, 126, 127, 129, 130, 132, 133, 134, 135, 136,
                137, 138, 140, 142, 144, 145, 146, 147, 148, 149, 150], 
                'val': [7, 8, 11, 15, 21, 23, 41, 48, 49, 55, 59, 62, 71, 81, 86, 109, 114, 128, 131, 141]}
    ids = dataset['train']
    for id in ids:
        DPTS(id,alpha=1)
