#!/usr/bin/env python3
import argparse
import tempfile
# import rawpy
from os.path import exists, splitext, abspath
from os import remove, environ
import SimpleITK as sitk

def read_raw(binary_file_name, image_size, sitk_pixel_type, image_spacing=None, image_origin=None, big_endian=False):
    """
    Read a raw binary scalar image.

    Parameters
    ----------
    binary_file_name (str): Raw, binary image file content.
    image_size (tuple like): Size of image (e.g. [2048,2048])
    sitk_pixel_type (SimpleITK pixel type: Pixel type of data (e.g. sitk.sitkUInt16).
    image_spacing (tuple like): Optional image spacing, if none given assumed to be [1]*dim.
    image_origin (tuple like): Optional image origin, if none given assumed to be [0]*dim.
    big_endian (bool): Optional byte order indicator, if True big endian, else little endian.

    Returns
    -------
    SimpleITK image or None if fails.
    """
    pixel_dict = {sitk.sitkUInt8: 'MET_UCHAR',
        sitk.sitkInt8: 'MET_CHAR',
        sitk.sitkUInt16: 'MET_USHORT',
        sitk.sitkInt16: 'MET_SHORT',
        sitk.sitkUInt32: 'MET_UINT',
        sitk.sitkInt32: 'MET_INT',
        sitk.sitkUInt64: 'MET_ULONG_LONG',
        sitk.sitkInt64: 'MET_LONG_LONG',
        sitk.sitkFloat32: 'MET_FLOAT',
        sitk.sitkFloat64: 'MET_DOUBLE'}

    direction_cosine = ['1 0 0 1', '1 0 0 0 1 0 0 0 1', '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1']

    dim = len(image_size)

    header = ['ObjectType = Image\n'.encode(), (f'NDims = {dim}\n').encode(), 
        ('DimSize = ' + ' '.join([str(v) for v in image_size]) + '\n').encode(),
        ('ElementSpacing = ' + (' '.join([str(v) for v in image_spacing]) if image_spacing else ' '.join(['1'] * dim)) + '\n').encode(), 
        ('Offset = ' + (' '.join([str(v) for v in image_origin]) if image_origin else ' '.join(['0'] * dim) + '\n')).encode(), 
        ('TransformMatrix = ' + direction_cosine[dim - 2] + '\n').encode(),
        ('ElementType = ' + pixel_dict[sitk_pixel_type] + '\n').encode(),
        'BinaryData = True\n'.encode(),
        ('BinaryDataByteOrderMSB = ' + str(big_endian) + '\n').encode(),
        # ElementDataFile must be the last entry in the header
        ('ElementDataFile = ' + abspath(binary_file_name) + '\n').encode()]

    fp = tempfile.NamedTemporaryFile(suffix='.mhd', delete=False)

    # Not using the tempfile with a context manager and auto-delete because on windows we can't open the file a second time for ReadImage.
    fp.writelines(header)
    fp.close()
    img = sitk.ReadImage(fp.name)
    remove(fp.name)

    return img # SimpleITK.SimpleITK.Image

if __name__=='__main__':
    psr = argparse.ArgumentParser(description='Read .raw binary image')
    psr.add_argument('--f', type=str, required=True, help='Path to the .raw file (raw binary image file)')
    psr.add_argument("--big_endian", type=lambda v: v.lower() in {"1", "true"}, required=True, help="\'false\' for little endian or \'true\' for big endian")
    psr.add_argument('--size', required=True, nargs='+', help="width, height", type=int)
    psr.add_argument('--type', default="sitkFloat32", help="SimpleITK pixel type (default: sitkFloat32)")

    args = psr.parse_args()
    
    if not exists(args.f):
        print(f"File {args.f} could not be found")
        exit(1)
    
    if splitext(args.f)[1].upper() !=  ".RAW":
        print(f"File extension must be .raw")
        exit(1)

    # with rawpy.imread(args.f) as raw:
    #     rgb = raw.postprocess()  

    string_to_pixelType = {"sitkUInt8": sitk.sitkUInt8,
        "sitkInt8": sitk.sitkInt8,
        "sitkUInt16": sitk.sitkUInt16,
        "sitkInt16": sitk.sitkInt16,
        "sitkUInt32": sitk.sitkUInt32,
        "sitkInt32": sitk.sitkInt32,
        "sitkUInt64": sitk.sitkUInt64,
        "sitkInt64": sitk.sitkInt64,
        "sitkFloat32": sitk.sitkFloat32,
        "sitkFloat64": sitk.sitkFloat64}

    # Read the image using both big and little endian
    image = read_raw(binary_file_name=args.f,
                    image_size=args.size,
                    sitk_pixel_type=string_to_pixelType[args.type],
                    big_endian=args.big_endian)
    
    print(f"H={image.GetHeight()}, W={image.GetWidth()}, D={image.GetDepth()}")

    out_file_name = splitext(args.f)[0] + ".tif"
    
    # save image
    sitk.WriteImage(image, out_file_name)

    # show image
    # sitk.Show(image, 'raw converted') # "SITK_NOSHOW" in os.environ