from pathlib import Path
from shutil import copyfile
import glob
import math

from medpy.filter import smoothing
from skimage import io, img_as_ubyte, exposure
import numpy as np
import cv2

def create_directory(path):
    path_to_file = Path(path)
    parent_directory_of_file = path_to_file.parent
    parent_directory_of_file.mkdir(parents=True, exist_ok=True)

def adaptive_gamma(r, g=None, b=None):
    # print(I_in)
    if g is None and b is None:
        I_in = r
    else:
        rgb_in = np.array([r, g, b]).transpose(1, 2, 0)
        hsv_in = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2HSV)
        h, s, I_in =  cv2.split(hsv_in)
    mu = np.mean(I_in)
    sigma = np.std(I_in)
    tau = 3.
    # D = np.diff(mu + 2. * sigma, mu - 2. * sigma)
    # rho = 1 if D <= 1 / tau else 2
    rho = 1 if 4 * sigma <= 1 / tau else 2
    bright = True if mu >= 0.5 else False
    dark = not bright

    if rho == 1:
        gamma = -math.log(sigma, 2)
    else:
        gamma = math.exp((1 - (mu + sigma)) / 2)

    k = (I_in ** gamma) + (1 - (I_in ** gamma)) * (mu ** gamma)
    Heaviside = lambda x: 0 if x <= 0 else 1
    c = 1 / (1 + Heaviside(0.5 - mu) * (k - 1))

    I_out = c * (I_in ** gamma)

    if g is None and b is None:
        return I_out
    else:
        hsv_out = cv2.merge([h, s, I_out])
        rgb_out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)
        return rgb_out

datasets_root = 'datasets'

input_dataset_name = 'eobrowser'
input_dataset_path = f'{datasets_root}/{input_dataset_name}'

output_dataset_name = 'eobrowser-processed'
output_dataset_path = f'{datasets_root}/{output_dataset_name}'

crop_size = (320, 320)

input_image_dir_pattern = f'{input_dataset_path}/32-*'

for input_image_dir in glob.glob(input_image_dir_pattern):
    output_image_dir = input_image_dir.replace(input_dataset_path, output_dataset_path)

    input_image_paths = sorted(glob.glob(f'{input_image_dir}/*.tiff'))
    for input_image_path in input_image_paths:
        if input_image_path.endswith('_NDSI.tiff'):
            continue

        output_image_path = input_image_path.replace(input_dataset_path, output_dataset_path)
        create_directory(output_image_path)

        if input_image_path.endswith('_(Raw).tiff'):
            copyfile(input_image_path, output_image_path)
            continue

        image = io.imread(input_image_path, plugin='tifffile')
        cropped_image = image[:crop_size[0], :crop_size[1]]
        output_image_path = output_image_path.replace('.tiff', '.png')
        io.imsave(output_image_path, img_as_ubyte(cropped_image))

        if output_image_path.endswith('_True_color.png'):
            via_image_path = '-'.join(output_image_path.replace(f'{output_dataset_path}/', '').split('/'))
            via_image_path = f'{output_dataset_path}/{via_image_path}'
            in_range = tuple(np.percentile(cropped_image, [2, 98]))
            enhanced_image = exposure.rescale_intensity(cropped_image, in_range=in_range, out_range=np.uint8)
            enhanced_image = smoothing.anisotropic_diffusion(enhanced_image) / 255.
            enhanced_image = adaptive_gamma(enhanced_image)
            # enhanced_image = img_as_ubyte(enhanced_image)
            # enhanced_image = cv2.medianBlur(enhanced_image, ksize=3)
            io.imsave(via_image_path, img_as_ubyte(enhanced_image))
