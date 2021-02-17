from argparse import ArgumentParser

import cv2


# Function to extract frames
def FrameCapture(args):
    vidcap = cv2.VideoCapture(args.video)
    success, image = vidcap.read()
    count = 0
    while success:
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
        cv2.imwrite("%s/%d.jpg" % (args.result_path, count), image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def parse_args():
    parser = ArgumentParser(description='Convert video to frames')
    parser.add_argument(
        '--video', required=True, type=str,
        help='Path to video'
    )
    parser.add_argument(
        '--result-path', required=True, type=str,
        help='Path to result frames'
    )
    return parser.parse_args()


if __name__ == '__main__':
    FrameCapture(parse_args())
