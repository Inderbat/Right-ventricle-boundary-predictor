import torch
import os
import pydicom
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from config import conf
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.exposure import equalize_adapthist


# generates the mask for a contour file
def generate_mask(file_path, image_size):
    mask_image = np.zeros(image_size, dtype=np.uint8)
    
    # parsing the contour file
    with open(file_path, 'r') as locations_file:
        for line in locations_file.readlines():
            line_parsed = line.strip('\n').split(' ')
            [x, y] = [float(num) for num in line_parsed]
            mask_image[int(y), int(x)] = 255
            
            # handling the case when the boundary is between the two pixels
            if int(y) + 1 < image_size[0]:
                mask_image[int(y) + 1, int(x)] = 255
            if int(x) + 1 < image_size[0]:
                mask_image[int(y), int(x) + 1] = 255
            if int(x) - 1 >= 0:
                mask_image[int(y), int(x) - 1] = 255
            if int(y) - 1 >= 0:
                mask_image[int(y) - 1, int(x)] = 255
            if int(y) + 1 < image_size[0] and int(x) + 1 < image_size[0]:
                mask_image[int(y) + 1, int(x) + 1] = 255
            if int(y) + 1 < image_size[0] and int(x) - 1 >= 0:
                mask_image[int(y) + 1, int(x) - 1] = 255
            if int(y) - 1 >= 0 and int(x) + 1 < image_size[0]:
                mask_image[int(y) - 1, int(x) + 1] = 255
            if int(y) - 1 >= 0 and int(x) - 1 >= 0:
                mask_image[int(y) - 1, int(x) - 1] = 255
                
    return mask_image

def generate_dataset(dir_name, target_dir_name='test_dir'):
    
    # filtering all the contour files
    all_files = os.listdir(dir_name)
    i_contour_files = list(filter(lambda x: 'icontour-manual.txt' in x, all_files))
    o_contour_files = list(filter(lambda x: 'ocontour-manual.txt' in x, all_files))
    assert(len(i_contour_files) == len(o_contour_files))
    dicom_image_files = [file_name[:8] + '.dcm' for file_name in i_contour_files]
    
    # creating a new directory
    if not os.path.isdir(target_dir_name):
        os.mkdir(target_dir_name)
    target_csv_file = target_dir_name + '.csv'
    target_dict = {'image_file': [], 'image_file_i_contour': [], 'image_file_o_contour': []}
    
    # generating images for each file
    for image_file in tqdm(dicom_image_files):
        image_ds = pydicom.dcmread(os.path.join(dir_name, image_file))
        image_name = os.path.splitext(image_file)[0]
        
        # saving the image
        image_pixel_array = image_ds.pixel_array

        # handling the Houston units to normal pixel convertion
        image_pixel_array[image_pixel_array > 1000] = 0
        image_pixel_array[image_pixel_array < -1000] = 0
        image_pixel_array = (image_pixel_array + 1000) * (255 / 2000)

        if image_pixel_array.shape[0] < image_pixel_array.shape[1]:
            img = Image.fromarray(image_pixel_array[::conf['down_sample'], ::conf['down_sample']].astype(np.uint8).T)
        else:
            img = Image.fromarray(image_pixel_array[::conf['down_sample'], ::conf['down_sample']].astype(np.uint8))
        img.save(os.path.join(target_dir_name, image_name + '.png'))
        
        # generating the masking contours
        mask_i_contour = generate_mask(os.path.join(dir_name, image_name + '-icontour-manual.txt'), image_pixel_array.shape)
        mask_o_contour = generate_mask(os.path.join(dir_name, image_name + '-ocontour-manual.txt'), image_pixel_array.shape)
        
        # handling the transpose case
        if image_pixel_array.shape[0] < image_pixel_array.shape[1]:
            mask_i_contour = mask_i_contour.T
            mask_o_contour = mask_o_contour.T
        
        # saving the masks
        mask_i = Image.fromarray(mask_i_contour[::conf['down_sample'], ::conf['down_sample']].astype(np.uint8))
        mask_o = Image.fromarray(mask_o_contour[::conf['down_sample'], ::conf['down_sample']].astype(np.uint8))
        mask_i.save(os.path.join(target_dir_name, image_name + '-icontour-manual.png'))
        mask_o.save(os.path.join(target_dir_name, image_name + '-ocontour-manual.png'))
        
        # updating the dictionary
        target_dict['image_file'].append(image_name + '.png')
        target_dict['image_file_i_contour'].append(image_name + '-icontour-manual.png')
        target_dict['image_file_o_contour'].append(image_name + '-ocontour-manual.png')
        
    # saving the dictionary
    target_df = pd.DataFrame.from_dict(target_dict)
    target_df.to_csv(os.path.join(target_dir_name, target_csv_file), index=False)

# for preparing the dataset object
class TensorToDataset(Dataset):
    def __init__(self, tensor_batch, target):
        self.tensor_batch = tensor_batch
        self.target = target

    def __len__(self):
        return len(self.tensor_batch)

    def __getitem__(self, item):
        tensor = self.tensor_batch[item]
        if self.target is None:
            return tensor, None
        return tensor, self.target[item]
    
# for preparing the dataloader object
def load_data(dir_name, dataset_split=0.2, clahe=True, supervision_level=1.0):
    
    # transforming each patches
    transform_obj = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # reading the csv file
    csv_file = pd.read_csv(os.path.join(dir_name, dir_name + '.csv'))
    
    # to save the images
    data = []
    masks = []
    
    # parsing the csv file
    for index in tqdm(csv_file.index):
        
        # saving the image
        image_data = np.array(Image.open(os.path.join(dir_name, csv_file['image_file'][index])))
        if clahe:
            image_data = equalize_adapthist(image_data)
            image_data = (image_data * 255).astype(np.uint8)
        data.append(transform_obj(image_data))
        
        # saving the label
        mask_i = (mpimg.imread(os.path.join(dir_name, csv_file['image_file_i_contour'][index])) > 0).astype(np.float)
        mask_o = (mpimg.imread(os.path.join(dir_name, csv_file['image_file_o_contour'][index])) > 0).astype(np.float)
        joint_mask = ((mask_i + mask_o) > 0).astype(np.float)
        masks.append(torch.from_numpy(joint_mask).view(1, *joint_mask.shape))

    # splitting on the basis of supervision level
    if supervision_level != 1.0:
        data, data_reject, masks, masks_reject = train_test_split(data, masks, test_size = 1 - supervision_level)
        
    # splitting the data into two datasets (train, cross validation)
    if dataset_split is not None:
        train_data, test_data, train_mask, test_mask = train_test_split(data, masks, test_size=dataset_split)
        dset_train = TensorToDataset(train_data, train_mask)
        dset_test = TensorToDataset(test_data, test_mask)
        data_loader_train = DataLoader(dset_train, batch_size=conf['batch_size'], shuffle=True, num_workers=conf['data_workers'])
        data_loader_test = DataLoader(dset_test, batch_size=conf['batch_size'], shuffle=True, num_workers=conf['data_workers'])
        return data_loader_train, data_loader_test
    else:
        dset = TensorToDataset(data, masks)
        data_loader = DataLoader(dset, batch_size=conf['batch_size'], shuffle=True, num_workers=conf['data_workers'])
        return data_loader
    
# data augmentation functions
def augment_training_data_loader(data_loader, rotation=True, translation=True, noise=True):
    
    # data augmentation parameters
    if rotation:
        angles = np.arange(-180, 181, 10)
    if noise:
        standard_deviations = [0.001, 0.01, 0.025]
    if translation:
        shifts = [-5, 0, +5]
        translations = []
        for x in shifts:
            for y in shifts:
                if x != 0 and y != 0:
                    translations.append((x, y))
        
    # to be returned
    data = []
    masks = []
    
    # data augmentation begins
    for batch_index, batch in tqdm(enumerate(data_loader)):
        for index in range(len(batch)):
            data.append(batch[0][index])
            masks.append(batch[1][index])
            image_data = torch.squeeze(batch[0][index]).numpy()
            mask_data = torch.squeeze(batch[1][index]).numpy()
            
            # applying the rotation transformation
            if rotation:
                for angle in angles:
                    if angle != 0:
                        transformed_image_data = rotate(image_data, angle=angle, mode='wrap')
                        transformed_mask_data = (rotate(mask_data, angle=angle, mode='wrap') > 0).astype(np.float)
                        data.append(torch.from_numpy(transformed_image_data).view(1, *transformed_image_data.shape))
                        masks.append(torch.from_numpy(transformed_mask_data).view(1, *transformed_mask_data.shape))
                        
            # applying the noise transformation
            if noise:
                for sd in standard_deviations:
                    transformed_image_data = random_noise(image_data, var=sd**2)
                    data.append(torch.from_numpy(transformed_image_data).view(1, *transformed_image_data.shape))
                    masks.append(batch[index][1])
                    
            # applying translations
            if translation:
                for shift in translations:
                    translator = AffineTransform(translation=shift)
                    transformed_image_data = warp(image_data, translator, mode='wrap')
                    transformed_mask_data = (warp(mask_data, translator, mode='wrap') > 0).astype(np.float)
                    data.append(torch.from_numpy(transformed_image_data).view(1, *transformed_image_data.shape))
                    masks.append(torch.from_numpy(transformed_mask_data).view(1, *transformed_mask_data.shape))
                    
    # creating the dataloader
    dset = TensorToDataset(data, masks)
    data_loader = DataLoader(dset, batch_size=conf['batch_size'], shuffle=True, num_workers=conf['data_workers'])
    
    return data_loader


if __name__ == '__main__':

    # parsing the command to format the dataset
    parser = argparse.ArgumentParser(description='Input among the following: train, test1, test2')
    parser.add_argument('--dataset', '-ds', required=True, type=str)

    args = parser.parse_args()
    if args.dataset == 'train':
        generate_dataset('TrainingSet', target_dir_name='train_dir')
    
