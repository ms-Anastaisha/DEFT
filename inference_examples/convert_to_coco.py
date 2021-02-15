from argparse import ArgumentParser
import os
import cv2
from PIL import Image
import numpy as np
import json
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
# from imantics import Mask


classes_names = [
    'shirt, blouse',
    'top, t-shirt, sweatshirt',
    'sweater',
    'cardigan',
    'jacket',
    'vest',
    'pants',
    'shorts',
    'skirt',
    'coat',
    'dress',
    'jumpsuit',
    'cape',
    'glasses',
    'hat',
    'headband, head covering, hair accessory',
    'tie',
    'glove',
    'watch',
    'belt',
    'leg warmer',
    'tights, stockings',
    'sock',
    'shoe',
    'bag, wallet',
    'scarf',
    'umbrella'
]

classes_wrap = [
    [0, 28, 30, 31, 33, 41],
    [1, 27, 30, 31, 33, 35, 41],
    [2, 30, 31, 35, 41],
    [3, 30, 31, 35, 41],
    [4, 27, 29, 30, 31, 35, 41],
    [5, 30, 31, 35],
    [6, 32, 42],
    [7, 32, 42],
    [8, 42],
    [9, 29, 30, 31, 35, 41],
    [10, 31, 32, 35, 37, 38, 39, 44],
    [11, 31],
    [12, 27],
    [13, 31, 33],
    [14],
    [15],
    [16],
    [17],
    [18],
    [19],
    [20],
    [21],
    [22],
    [23, 34],
    [24],
    [25],
    [26],
]


def get_real_class(cl: int, classes: list) -> int:
    global classes_wrap
    for idx, class_wrap in enumerate(classes_wrap):
        if cl in class_wrap and classes_wrap[idx][0] in classes:
            return idx
    return -1


def create_mask(df: pd.DataFrame):
    num_categories = 45 + 1  # (add 1 for background)

    mask_h = df.loc[:, 'Height'][0]
    mask_w = df.loc[:, 'Width'][0]
    mask = np.full(mask_w * mask_h, num_categories - 1, dtype=np.int32)

    for encode_pixels, encode_labels in zip(df.EncodedPixels.values,
                                            df.ClassId.values):
        pixels = list(map(int, encode_pixels.split(' ')))
        for i in range(0, len(pixels), 2):
            start_pixel = pixels[i] - 1  # index from 0
            len_mask = pixels[i + 1] - 1
            end_pixel = start_pixel + len_mask
            if int(encode_labels) < num_categories - 1:
                mask[start_pixel:end_pixel] = get_real_class(
                    int(encode_labels),
                    df.ClassId.values
                )

    mask = mask.reshape((mask_h, mask_w), order='F')
    return mask


def imap_unordered_bar(func, args, n_processes=8):
    p = Pool(n_processes)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def init_coco_dict() -> dict:
    return {
        "info": {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc/2.0/",
                "id": 2,
                "name": "Attribution-NonCommercial License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
                "id": 3,
                "name": "Attribution-NonCommercial-NoDerivs License"
            },
            {
                "url": "http://creativecommons.org/licenses/by/2.0/",
                "id": 4,
                "name": "Attribution License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-sa/2.0/",
                "id": 5,
                "name": "Attribution-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nd/2.0/",
                "id": 6,
                "name": "Attribution-NoDerivs License"
            },
            {
                "url": "http://flickr.com/commons/usage/",
                "id": 7,
                "name": "No known copyright restrictions"
            },
            {
                "url": "http://www.usa.gov/copyright.shtml",
                "id": 8,
                "name": "United States Government Work"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }


def get_image_description(idx: int, image: Image) -> dict:
    return {
        "license": 1,
        "file_name": '{}.jpg'.format(idx),
        "coco_url": '',
        "height": image.size[1],
        "width": image.size[0],
        "date_captured": '',
        "flickr_url": '',
        "id": idx
    }


def divide_mask(mask: np.ndarray) -> list:
    num_labels, labels_mask = cv2.connectedComponents(mask)
    if num_labels != 3:
        return [None]
    mask_left = np.copy(labels_mask)
    mask_right = np.copy(labels_mask)

    mask_left[mask_left == 1] = 0
    mask_right[mask_right == 2] = 0

    mask_left[mask_left > 0] = 255
    mask_right[mask_right > 0] = 255

    return [
        mask_left.astype(np.uint8),
        mask_right.astype(np.uint8)
    ]


def worker_mask_generation_function(arg):
    idx, df, image_path, save_image_path = arg
    try:
        mask = create_mask(df)
    except Exception as e:
        print(e)
        print('Skipped index: {}'.format(idx))
        return [idx, None, None]

    indexes = list(
        set(
            [
                ind
                for ind in set(df.ClassId.to_list())
                if ind < 27
            ]
        )
    )

    if len(indexes) > 0:
        masks = [
            (mask == class_index).astype(np.uint8) * 255
            for class_index in indexes
        ]
    else:
        print('Skipped index: {}'.format(idx))
        return [idx, None, None]

    result = []

    image = Image.open(image_path)
    w, h = image.size

    for i, m in enumerate(masks):
        if classes_names[indexes[i]] in ['shoe', 'sock', 'tights, stockings']:
            separated_masks = divide_mask(m)

            if separated_masks[0] is not None:
                for sm in separated_masks:
                    wrapped_mask = Mask(sm)

                    area = wrapped_mask.area()
                    segmentation_data = wrapped_mask.polygons().segmentation
                    box = cv2.boundingRect(sm)

                    segmentation_data = [
                        np.array(segm_elem).reshape((-1, 2))[::5].astype(
                            np.float32).flatten().tolist()
                        for segm_elem in segmentation_data
                        if len(segm_elem) >= 7 * 5
                    ]

                    if len(segmentation_data) == 0:
                        continue

                    if box[0] < 5 or box[1] < 5:
                        continue

                    if box[2] * box[3] < 150:
                        continue

                    result.append(
                        {
                            'segmentation': segmentation_data,
                            'bbox': box,
                            'area': float(area),
                            'iscrowd': 0,
                            'category_id': indexes[i] + 1,
                            'image_id': idx,
                            'id': -1
                        }
                    )
                continue

        wrapped_mask = Mask(masks[i])

        area = wrapped_mask.area()
        segmentation_data = wrapped_mask.polygons().segmentation
        box = cv2.boundingRect(m)

        segmentation_data = [
            np.array(segm_elem).reshape((-1, 2))[::5].astype(
                np.float32).flatten().tolist()
            for segm_elem in segmentation_data
            if len(segm_elem) >= 7*5
        ]

        if len(segmentation_data) == 0:
            continue

        if box[0] < 5 or box[1] < 5:
            continue

        if box[2]*box[3] < 150:
            continue

        # try:
        #     rles = coco.maskUtils.frPyObjects(segmentation_data, h, w)
        # except Exception as e:
        #     print(e)
        #     print('On mask {}'.format(segmentation_data))
        #     print('Len: {}'.format(len(segmentation_data)))
        #     for sd in segmentation_data:
        #         print('Part len: {}'.format(len(sd)))
        #     raise RuntimeError()

        result.append(
            {
                'segmentation': segmentation_data,
                'bbox': box,
                'area': float(area),
                'iscrowd': 0,
                'category_id': indexes[i] + 1,
                'image_id': idx,
                'id': -1
            }
        )

    if len(result) == 0:
        print('Skipped index: {}'.format(idx))
        return [idx, None, None]

    image_description = get_image_description(idx, image)
    image.save(save_image_path)

    return [idx, result, image_description]


def parse_args():
    parser = ArgumentParser(description='DeepFashion2 dataset converter')
    parser.add_argument(
        '--imaterialistic-path', required=True, type=str,
        help='Path to iMaterialistic Fashion dataset root dir part.'
    )
    parser.add_argument(
        '--result-coco-path', required=True, type=str,
        help='Path to created dataset root dir in COCO like type.'
    )
    parser.add_argument(
        '--val-part', required=False, type=float, default=0.1
    )
    parser.add_argument(
        '--njobs', required=False, type=int, default=24
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root_path = args.imaterialistic_path
    images_path = os.path.join(root_path, 'train/')
    images_names = os.listdir(images_path)

    with open(
            os.path.join(root_path, 'checked_names.txt'),
            'r'
    ) as f:
        masks_names = [line.rstrip() for line in f]

    assert len(images_names) >= len(masks_names)
    assert 0 <= args.val_part <= 1

    if not os.path.isdir(args.result_coco_path):
        os.makedirs(args.result_coco_path)

    if not os.path.isdir(os.path.join(args.result_coco_path, 'images/')):
        os.makedirs(os.path.join(args.result_coco_path, 'images/'))

    if not os.path.isdir(os.path.join(args.result_coco_path, 'annotations/')):
        os.makedirs(os.path.join(args.result_coco_path, 'annotations/'))

    n = int(len(masks_names) * (1 - args.val_part))

    train_masks_names = masks_names[:n]
    test_masks_names = masks_names[n:]

    train_csv = pd.read_csv(
        os.path.join(root_path, 'train.csv'),
        index_col=['ImageId']
    )

    train_coco = init_coco_dict()
    val_coco = init_coco_dict()

    common_image_index = 1

    print('############### TRAIN PART ###############')
    print('Configure train data for processing')
    masks_building_tasks_data = []

    for name in tqdm(train_masks_names):
        idx = common_image_index
        image_path = os.path.join(images_path, '{}.jpg'.format(name))
        save_image_path = os.path.join(
            args.result_coco_path,
            'images/',
            '{}.jpg'.format(idx)
        )

        df = train_csv.loc[name]
        if "Series" in str(type(df)):
            df = pd.DataFrame(
                [df.to_list()],
                columns=[
                    'EncodedPixels',
                    'Height',
                    'Width',
                    'ClassId',
                    'AttributesIds'
                ]
            )

        masks_building_tasks_data.append(
            [idx, df.copy(deep=True), image_path, save_image_path]
        )

        common_image_index += 1

    print('Configure train data for processing')
    masks_building_tasks_results = imap_unordered_bar(
        worker_mask_generation_function,
        masks_building_tasks_data,
        args.njobs
    )

    masks_building_tasks_results.sort(key=lambda x: x[0])

    object_index = 1

    print('Building train data')
    for elem in tqdm(masks_building_tasks_results):
        idx, objects_data, image_description = elem
        if objects_data is None:
            continue

        if len(objects_data) == 0:
            continue

        train_coco['images'].append(image_description)

        for obj_elem in objects_data:
            train_coco['annotations'].append(obj_elem)
            train_coco['annotations'][-1]['id'] = object_index

            train_coco['categories'].append(
                {
                    'id': object_index,
                    'name': classes_names[obj_elem['category_id'] - 1],
                    'supercategory': 'clothes'
                }
            )

            object_index += 1

    with open(
            os.path.join(
                args.result_coco_path,
                'annotations/',
                'instances_train2017.json'
            ),
            'w'
    ) as jf:
        json.dump(train_coco, jf)

    print('Train dataset part saved')

    print('############### VALIDATION PART ###############')
    print('Configure validation data for processing')
    masks_building_tasks_data = []

    for name in tqdm(test_masks_names):
        idx = common_image_index
        image_path = os.path.join(images_path, '{}.jpg'.format(name))
        save_image_path = os.path.join(
            args.result_coco_path,
            'images/',
            '{}.jpg'.format(idx)
        )

        df = train_csv.loc[name]
        if "Series" in str(type(df)):
            df = pd.DataFrame(
                [df.to_list()],
                columns=[
                    'EncodedPixels',
                    'Height',
                    'Width',
                    'ClassId',
                    'AttributesIds'
                ]
            )

        masks_building_tasks_data.append(
            [idx, df.copy(deep=True), image_path, save_image_path]
        )

        common_image_index += 1

    print('Configure validation data for processing')
    masks_building_tasks_results = imap_unordered_bar(
        worker_mask_generation_function,
        masks_building_tasks_data,
        args.njobs
    )

    masks_building_tasks_results.sort(key=lambda x: x[0])

    object_index = 1

    print('Building validation data')
    for elem in tqdm(masks_building_tasks_results):
        idx, objects_data, image_description = elem
        if objects_data is None:
            continue

        if len(objects_data) == 0:
            continue

        val_coco['images'].append(image_description)

        for obj_elem in objects_data:
            val_coco['annotations'].append(obj_elem)
            val_coco['annotations'][-1]['id'] = object_index

            val_coco['categories'].append(
                {
                    'id': object_index,
                    'name': classes_names[obj_elem['category_id'] - 1],
                    'supercategory': 'clothes'
                }
            )

            object_index += 1

    with open(
            os.path.join(
                args.result_coco_path,
                'annotations/',
                'instances_val2017.json'
            ),
            'w'
    ) as jf:
        json.dump(val_coco, jf)

    print('Validation dataset part saved')


if __name__ == '__main__':
    main()
