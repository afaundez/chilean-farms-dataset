from pathlib import Path
from shutil import copyfile
from skimage import io, img_as_ubyte, exposure
import glob
import numpy as np

def create_directory(path):
    path_to_file = Path(path)
    parent_directory_of_file = path_to_file.parent
    parent_directory_of_file.mkdir(parents=True, exist_ok=True)

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
            io.imsave(via_image_path, enhanced_image)
