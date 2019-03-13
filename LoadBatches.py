import time
import numpy as np
import openslide
import cv2
import random
from imgaug import augmenters as iaa

from keras.utils import to_categorical

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential([
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                iaa.Flipud(0.5),
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.WithColorspace(
                    to_colorspace="HSV",
                    from_colorspace="RGB",
                    children=iaa.WithChannels(0, iaa.Add((0, 15)))
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.CropAndPad(
                        percent=(0, 0.2),
                        pad_mode=["constant"],
                        pad_cval=(0, 128)
                    )
                ),
                iaa.ContrastNormalization((0.5, 1.5)),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    rotate=(-45, 45)
                )
            ])


def getimg(number, x, y, width=244, height=244):
    reader = openslide.OpenSlide(f'/home/data/ACDC/Images/{number}.tif')

    try:
        img = reader.read_region((y, x), 0, (width, height)).convert('RGB')
        img = np.asarray(img)

        assert img.shape == (width, height, 3)
        reader.close()
        return img
    except:
        pass
        # print(f'error tif image{number}')
def get(number, x, y, width=244, height=244, mask_size = 120):
    reader = openslide.OpenSlide(f'/home/data/ACDC/Images/{number}.tif')
    reader2 = openslide.OpenSlide(f'/home/data/ACDC/Mask/{number}_mask.tif')
    try:
        img = reader.read_region((y, x), 0, (width, height)).convert('RGB')
        img = np.array(img)
        mask = reader2.read_region((y, x), 0, (width, height)).convert('RGB')
        mask = np.array(mask)[:,:,0]
        offset = int((width-mask_size*2)/2)
        mask = mask[offset:mask_size*2+offset,offset:mask_size*2+offset]
        mask = cv2.resize(mask, (mask_size,mask_size))
        assert img.shape == (width, height, 3)
       
        reader.close()
        reader2.close()
        return [img, mask]
    except Exception as ex:
        print(ex)
        pass
        # print(f'error tif image{number}')

def DataGenerator(flag, model_size, batch_size, n_classes, arg, verbose=False, patch_number=4):
    i = 0
    X = []
    Y = []
    if flag=='val':
        with open(f'/home/data/ACDC/Patch{patch_number}/' + flag + '_0.txt', 'r') as f:
            list0 = f.readlines()
            random.shuffle(list0)
        with open(f'/home/data/ACDC/Patch{patch_number}/' + flag + '_1.txt', 'r') as f:
            list1 = f.readlines()
            random.shuffle(list1)
    else:
        with open('hard_train_0.txt', 'r') as f:
            list0 = f.readlines()
            random.shuffle(list0)
        with open('hard_train_1.txt', 'r') as f:
            list1 = f.readlines()
            random.shuffle(list1)     
    _list = [list0, list1]
    k = np.zeros((2,))
    while True:
        if i == batch_size:
            if np.array(X).shape[0] == batch_size and np.array(Y).shape[0] == batch_size:
                X = np.array(X)
                Y = np.array(Y)
                if arg:
                    X = seq.augment_images(X)
                X = np.asarray(X, dtype=np.float64)/255.
                #print(X.shape,Y.shape)
                yield X, Y
                #break
            i = 0
            X = []
            Y = []

        label = random.randint(0, 1)
        t_label = np.zeros((1, n_classes)).astype('uint')
        t_label[0, label] = 1
        while True:
            l = _list[label][int(k[label])]
            k[label]+=1
            if k[label]==len(_list[label]):
                k[label]=0
            number = int(l.split(',')[0])
            x = int(l.split(',')[1])
            y = int(l.split(',')[2])
            size = int(l.split(',')[3])
            try:
                img = getimg(number, x, y, width=size, height=size)
                img = cv2.resize(img, (model_size,model_size))  
                break            
            except:
                print('error!',l)
                continue
        if verbose:
            pass
        X.append(img)
        Y.append(t_label)
        i += 1
def seg_DataGenerator(flag, model_size, batch_size, n_classes, arg, mask_size, verbose=False):
    i = 0
    X = []
    Y = []
    Y2 = []
    with open('/home/data/ACDC/Patch4/' + flag + '_0.txt', 'r') as f:
        list0 = f.readlines()
        random.shuffle(list0)
    with open('/home/data/ACDC/Patch4/' + flag + '_1.txt', 'r') as f:
        list1 = f.readlines()
        random.shuffle(list1)
    _list = [list0, list1]
    k = np.zeros((2,))
    while True:
        if i == batch_size:
            if np.array(X).shape[0] == batch_size and np.array(Y).shape[0] == batch_size:
                X = np.array(X)
                Y = np.array(Y)
                Y2 = np.array(Y2)
                if arg:
                    seq_det = seq.deterministic()
                    X = seq_det.augment_images(X)
                    Y2 = seq_det.augment_images(Y2)
                X = np.asarray(X, dtype=np.float64)/255.
                Y2 = np.reshape(Y2, (batch_size, -1))
                Y2 = to_categorical(Y2, num_classes=n_classes)
                yield X, [Y,Y2]
            i = 0
            X = []
            Y = []
            Y2 = []
        label = random.randint(0, 1)
        t_label = np.zeros((1, n_classes)).astype('uint')
        t_label[0, label] = 1
        while True:
            l = _list[label][int(k[label])]
            k[label]+=1
            if k[label]==len(_list[label]):
                k[label]=0
            number = int(l.split(',')[0])
            x = int(l.split(',')[1])
            y = int(l.split(',')[2])
            size = int(l.split(',')[3])
            #[img, mask] = get(number, x, y, width=size, height=size, mask_size=mask_size)
            # if not img.shape[0]==model_size:
            #     img = cv2.resize(img, (model_size,model_size))  
            # break            
            try:
                img, mask = get(number, x, y, width=size, height=size, mask_size=mask_size)
                if not img.shape[0]==model_size:
                    img = cv2.resize(img, (model_size,model_size))  
                break            
            except Exception as ex:
                print(ex,l)
                continue
        if verbose:
            print(l)
        X.append(img)
        Y.append(t_label)
        Y2.append(mask)
        i += 1

if __name__ == "__main__":
    for i,(X,Y) in enumerate(DataGenerator('train', 244, 16, 4, False, verbose=True)):
        for k in range(X.shape[0]):
            #print(Y2[:,1800:1810])
            label = np.where(Y[k][0]==1)[0][0]
            cv2.imwrite(f'testgener/{i}_{k}_{label}.jpg', X[k]*255)
            # cv2.imwrite(f'testgener/{i}_{k}_{label}_mask.jpg',Y2[k]*255)
        
        if i==4:
            break
    # get(148,140687, 48392)
