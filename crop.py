from pathlib import Path
from shutil import copyfile
from skimage import io, img_as_ubyte
import glob

def create_directory(path):
    path_to_file = Path(path)
    parent_directory_of_file = path_to_file.parent
    parent_directory_of_file.mkdir(parents=True, exist_ok=True)

input_root = 'input'
output_root = 'output'
crop_size = (320, 320)

image_dir_pattern = f'{input_root}/32-*'

for input_image_dir in glob.glob(image_dir_pattern):
    output_image_dir = input_image_dir.replace('input', 'output')
    input_image_paths = sorted(glob.glob(f'{input_image_dir}/*.tiff'))
    for input_image_path in input_image_paths:
        if input_image_path.endswith('_NDSI.tiff'):
            continue

        output_path = f'{output_image_dir}/{input_image_path.split("/")[-1]}'
        create_directory(output_path)

        if input_image_path.endswith('_(Raw).tiff'):
            copyfile(input_image_path, output_path)
            continue

        image = io.imread(input_image_path, plugin='tifffile')
        cropped_image = image[:crop_size[0], :crop_size[1]]
        output_path = output_path.replace('.tiff', '.png')
        io.imsave(output_path, img_as_ubyte(cropped_image))
