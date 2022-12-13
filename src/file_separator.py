from xml.etree import ElementTree as et
import glob as glob
import shutil

files = glob.glob('C:\\Users\\AatishNehe\\Downloads\\Helipad_Detection\\Data\\train\\*.xml')

destination_folder = 'C:\\Users\\AatishNehe\\Downloads\\Helipad_Detection\\Objects\\train\\'

count = 0

for file in files:
    tree = et.parse(file)
    root = tree.getroot()
    members = root.findall('object')
    count += 1
    orig_dir = 'C:\\Users\\AatishNehe\\Downloads\\Helipad_Detection\\Data\\train\\'

    if not len(members) == 0:
        destination = destination_folder + file.split('\\')[-1]
        destination_image = destination[:-4] + '.jpg'
        orig_image = orig_dir + destination_image.split('\\')[-1]
        shutil.copy(file, destination)
        shutil.copy(orig_image, destination_image)
