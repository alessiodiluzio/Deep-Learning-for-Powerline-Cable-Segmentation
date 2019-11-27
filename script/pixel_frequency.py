from PIL import Image
from src.utils import get_images
IMAGE_PATH = "../file/input/label"


def pixel_frequency(training_images_label):
    white_count = 0
    black_count = 0
    total_pixel = 0
    i = 0
    for img in training_images_label:
        i += 1
        img = Image.open(img)
        pix = list(img.getdata())
        total_pixel += len(pix)
        black_count += pix.count(0)
        white_count += len(pix) - black_count
    perc_black = float(black_count) / total_pixel
    perc_white = 1 - perc_black
    print("W : ", perc_white, "\nB : ", perc_black)
    return perc_black, perc_white


def write_pixel_frequency(file_path):
    training_images = get_images(IMAGE_PATH)
    perc_black, perc_white = pixel_frequency(training_images)
    with open(file_path, 'w') as file:
        line_1 = "WHITE=" + str(perc_white) + '\n'
        line_2 = "BLACK=" + str(perc_black)
        file.write(line_1)
        file.write(line_2)


write_pixel_frequency('../file/WHITE_BLACK_PERCENTUAL.txt')
