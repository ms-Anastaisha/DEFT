import os
import json
from argparse import ArgumentParser


def init_coco_dict() -> dict:
    return {

        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "pedestrian"
            }
        ],
        "videos": []
    }


def parse_args():
    parser = ArgumentParser(description='Convert video to coco format')
    parser.add_argument(
        '--data-path', required=True, type=str,
        help='Path to images'
    )
    parser.add_argument(
        '--out-path', required=True, type=str,
        help='Path to coco annotation'
    )
    return parser.parse_args()


def video2coco(args):
    input_path = args.data_path
    out_path = args.out_path
    out = init_coco_dict()
    out["videos"].append({"id": 1, "file_name": input_path})
    images = os.listdir(input_path)
    num_images = len([image for image in images if "jpg" in image])
    for i in range(num_images):
        image_info = {
            "file_name": images[i],
            "id": i + 1,
            "frame_id": i + 1,
            "prev_image_id": i if i > 0 else -1,
            "next_image_id": i + 2 if i < num_images - 1 else -1,
            "video_id": out["videos"][0]["id"]
        }
        out["images"].append(image_info)
    json.dump(out, open(out_path, 'w'))


if __name__ == '__main__':
    video2coco(parse_args())
