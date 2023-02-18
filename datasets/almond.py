###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to create the Street View House Numbers (SVHN) Dataset.
(http://ufldl.stanford.edu/housenumbers/)
Format: 1 is used: Format with Bounding Boxes
"""
import ast
import errno
import os
import pickle
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import h5py
import pandas as pd
from PIL import Image

import ai8x


class ALMOND(Dataset):
    """
    Street View House Numbers (SVHN) Dataset. (http://ufldl.stanford.edu/housenumbers/)
    Format: 1 is used: Format with Bounding Boxes

    Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits
    in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and
    Unsupervised Feature Learning 2011.
    """

    expansion_ratio = 0.3

    def __init__(self, root_dir, d_type, transform=None, img_size=(168, 224), fold_ratio=2,
                 simplified=False):

        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        print("IMAGES ARE EXPECTED TO BE 168x224 !")
        self.root_dir = root_dir
        self.d_type = d_type
        self.transform = transform
        self.img_size = img_size
        self.fold_ratio = fold_ratio
        self.simplified = simplified

        self.img_list = []
        self.boxes_list = []
        self.lbls_list = []

        self.processed_folder = os.path.join(root_dir, self.__class__.__name__, 'processed')
        self.__makedir_exist_ok(self.processed_folder)

        res_string = str(self.img_size[0]) + 'x' + str(self.img_size[1])
        simplified_string = "_simplified" if self.simplified else ""

        train_pkl_file_path = os.path.join(self.processed_folder, 'train_' + res_string +
                                           '_fold_' + str(self.fold_ratio) + simplified_string +
                                           '.pkl')
        test_pkl_file_path = os.path.join(self.processed_folder, 'test_' + res_string + '_fold_' +
                                          str(self.fold_ratio) + simplified_string + '.pkl')

        if self.d_type == 'train':
            self.pkl_file = train_pkl_file_path
            self.info_df_csv_file = os.path.join(self.processed_folder, 'train_info.csv')
        elif self.d_type == 'test':
            self.pkl_file = test_pkl_file_path
            self.info_df_csv_file = os.path.join(self.processed_folder, 'test_info.csv')
        else:
            print(f'Unknown data type: {self.d_type}')
            return

        self.__create_info_df_csv()
        self.__create_pkl_file()
        self.is_truncated = False

    def __create_info_df_csv(self):

        if os.path.exists(self.info_df_csv_file):
            self.info_df = pd.read_csv(self.info_df_csv_file)

            for column in self.info_df.columns:
                if column in ['label', 'x0', 'x1', 'y0', 'y1']:
                    self.info_df[column] = \
                        self.info_df[column].apply(ast.literal_eval)
        else:
          print("cannot find csv info files!")
          exit(0)
            

    def __create_pkl_file(self):

        if os.path.exists(self.pkl_file):

            (self.img_list, self.boxes_list, self.lbls_list) = \
                    pickle.load(open(self.pkl_file, 'rb'))
            return
        self.__gen_datasets()

    def __gen_datasets(self):
        print('\nGenerating dataset pickle file from the raw image files...\n')

        total_num_of_processed_files = 0

        for _, row in self.info_df.iterrows():
            num_box = row['num_of_boxes']
            #if num_box > 1:
              #print(f'num_box = {num_box}, drop!')
            #  continue
              
            if row['bb_y0'] <0:
              row['bb_y0'] = 0
            if row['bb_x0'] <0:
              row['bb_x0'] = 0            
            
            # print(row)
            img_width = row['img_width']
            img_height = row['img_height']

            # Read image
            image = Image.open(os.path.join(self.root_dir, self.__class__.__name__, self.d_type,
                                            row['img_name']))
            #image.show()
            # Crop expanded square first:
            img_crp = image

            #img_crp.show()
            # Resize cropped expanded square:
            img_crp_resized = img_crp
            #img_crp_resized.show()
            img_crp_resized.save('temp.jpg')#(row['img_name'][])

            img_crp_resized = np.asarray(img_crp_resized).astype(np.uint8)


            # Fold cropped expanded square (96 x 96 x 3 folded into 48 x 48 x 12) if required:
            img_crp_resized_folded = self.fold_image(img_crp_resized, self.fold_ratio)

            self.img_list.append(img_crp_resized_folded)

            boxes = []

            for b in range(len(row['x0'])):

                # Adjust boxes' coordinates wrt cropped image:
                x0_new = row['x0'][b]
                y0_new = row['y0'][b]
                x1_new = row['x1'][b]
                y1_new = row['y1'][b]


                boxes.append([x0_new, y0_new, x1_new, y1_new])

                #print(x0_new, y0_new, x1_new, y1_new)

            self.boxes_list.append(boxes)

            lbls = row['label']

            if self.simplified:
                # All boxes will have label 1 in simplified version, instead of digit labels
                lbls = [1] * len(lbls)

            self.lbls_list.append(lbls)

            total_num_of_processed_files = total_num_of_processed_files + 1

        # Save pickle file in memory
        pickle.dump((self.img_list, self.boxes_list, self.lbls_list), open(self.pkl_file, 'wb'))

        print(f'\nTotal number of processed files: {total_num_of_processed_files}\n')

    def __len__(self):
        if self.is_truncated:
            return 1
        return len(self.img_list)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        if self.is_truncated:
            index = 0

        if torch.is_tensor(index):
            index = index.tolist()

        img = self.img_list[index]
        boxes = self.boxes_list[index]
        lbls = self.lbls_list[index]

        img = self.__normalize_image(img).astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)
            #print(boxes)
            # Normalize boxes:
            boxes = [[box_coord / self.img_size[i%2] for i,box_coord in enumerate(box)] for box in boxes]
            #print(boxes)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(lbls, dtype=torch.int64)
            
            #print(img.shape)
            #exit(0)

        return img, (boxes, labels)

    @staticmethod
    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function
        (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
        """
        images = []
        boxes_and_labels = []

        for b in batch:
            images.append(b[0])
            boxes_and_labels.append(b[1])

        images = torch.stack(images, dim=0)
        return images, boxes_and_labels

    @staticmethod
    def get_name(index, hdf5_data):
        """Retrieve name field from hdf5 data"""
        name = hdf5_data['/digitStruct/name']
        return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])

    @staticmethod
    def get_bbox(index, hdf5_data):
        """Retrieve bounding box field from hdf5 data"""
        attrs = pd.DataFrame()
        item = hdf5_data['digitStruct/bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = hdf5_data[item][key]
            values = [int(hdf5_data[attr[()][i].item()][()][0][0])
                      for i in range(len(attr))] if len(attr) > 1 else [attr[()][0][0]]
            attrs[key] = values
        # Rename 'left' to 'x0'
        attrs['x0'] = attrs.pop('left')
        # Rename 'top' to 'y0'
        attrs['y0'] = attrs.pop('top')

        return attrs


    @staticmethod
    def get_image_size(image_path):
        """Returns image dimensions
        """
        image = Image.open(image_path)
        return image.size

    @staticmethod
    def __normalize_image(image):
        """Normalizes RGB images
        """
        return image / 256

    @staticmethod
    def __makedir_exist_ok(dirpath):
        """Make directory if not already exists
        """
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    @staticmethod
    def fold_image(img, fold_ratio):
        """Folds high resolution H-W-3 image h-w-c such that H * W * 3 = h * w * c.
           These correspond to c/3 downsampled images of the original high resoluiton image."""
        if fold_ratio == 1:
            img_folded = img
        else:
            img_folded = None
            for i in range(fold_ratio):
                for j in range(fold_ratio):
                    if img_folded is not None:
                        img_folded = \
                            np.concatenate((img_folded, img[i::fold_ratio,
                                                            j::fold_ratio, :]), axis=2)
                    else:
                        img_folded = img[i::fold_ratio, j::fold_ratio, :]
        return img_folded


def ALMOND_get_datasets(data, load_train=True, load_test=True, img_size=(168, 224), fold_ratio=2,
                      simplified=False):

    """ Returns SVHN Dataset
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = ALMOND(root_dir=data_dir, d_type='train',
                             transform=train_transform, img_size=img_size,
                             fold_ratio=fold_ratio, simplified=simplified)

        print(f'Train dataset length: {len(train_dataset)}\n')
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             ai8x.normalize(args=args)])

        test_dataset = ALMOND(root_dir=data_dir, d_type='test',
                            transform=test_transform, img_size=img_size,
                            fold_ratio=fold_ratio, simplified=simplified)

        print(f'Test dataset length: {len(test_dataset)}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset


def ALMOND_168_224_get_datasets(data, load_train=True, load_test=True):
    """ Returns SVHN Dataset with 96x96 images and simplified labels: 1 for every digit
    """
    return ALMOND_get_datasets(data, load_train, load_test, img_size=(168, 224), fold_ratio=1,
                             simplified=True)

datasets = [
   {
       'name': 'ALMOND',
       'input': (3, 224, 168),
       'output': ([1]),
       'loader': ALMOND_168_224_get_datasets,
       'collate': ALMOND.collate_fn
   }
]

#SVHN_168_224_get_datasets(('data', '') , load_train=True, load_test=True)
