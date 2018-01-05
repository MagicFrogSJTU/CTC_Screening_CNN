# Created by Chen Yizhi. 20171107.
# func:
#      transform3D()
#

import numpy as np
import SimpleITK as sitk
import time

def transform3D(vol, afa=0, beta=0, output_size=None, origin=None, zoom_ratio=1,
                    defaultPixelValue=0, interp_style='linear'):
    """Rotate, zoom, and translate volume.
    The center of rotation and zooming is the origin(Index in the coordinate of input volume), also as
    the center of the output volume.
    Args:
        vol: input. Must be 3D numpy array.
        afa: degree of rotation along axis Z. 0 ~ 2pi
        beta: degree of rotation along axis X. 0 ~ 2pi
        output_size: the size of the output. Must be 1D array of length 3.
        origin:  the center of the output as in the coordinate of input. Default is the center of input.
        zoom_ratio: the ratio of zooming. must be a single number. Bigger ratio, bigger original image.
        defaultPixelValue: default value of pixels outside of volume.
        interp_style: linear, nearest, B-Spline, or Gaussian.
    """

    if output_size is None:
        output_size = vol.shape
    else:
        output_size = np.array(output_size).astype(np.int)

    if interp_style == 'linear':
        interpolator = sitk.sitkLinear
    elif interp_style == 'nearest':
        interpolator = sitk.sitkLinear
    elif interp_style == 'B-Spline':
        interpolator = sitk.sitkBSpline
    elif interp_style == 'Gaussian':
        interpolator = sitk.sitkGaussian
    else:
        raise NameError

    input = sitk.GetImageFromArray(vol)

    if origin is None:
        origin = np.array(vol.shape)/2
    else:
        origin = np.array(origin)
    output_origin = origin.astype(np.float64)

    transform = sitk.VersorRigid3DTransform()
    transform.SetCenter(output_origin)
    transform.SetRotation([1, 0, 0], afa)
    transform.SetRotation([0, 1, 0], beta)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(output_origin-output_size*zoom_ratio/2)
    resampler.SetOutputSpacing((zoom_ratio, zoom_ratio, zoom_ratio))
    resampler.SetSize(output_size)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(defaultPixelValue)
    resampler.SetTransform(transform)

    output = resampler.Execute(input)
    return sitk.GetArrayFromImage(output)

'''
def _rotate_array(array, afa, beta):
    cos_afa = np.cos(afa)
    sin_afa = np.sin(afa)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)

    input = np.array(array).astype(np.float64)
    output = input.copy()

    output[0] = input[0] * cos_afa - input[1] * sin_afa
    output[1] = input[0] * sin_afa + input[1] * cos_afa

    input = output.copy()

    output[1] = input[1] * cos_beta - input[2] * sin_beta
    output[2] = input[1] * sin_beta + input[2] * cos_beta

    return output
'''
if __name__ == '__main__':
    input = sitk.ReadImage("L:\\testFiles\\vol.nii.gz")
    print(input.GetDirection())
    print(input.GetOrigin())
    vol = sitk.GetArrayFromImage(input)
    i = 45
    j = 45
    afa = i / 180.0 * np.pi
    beta = j / 180.0 * np.pi
    size = 32
    zoom_ratio = 1

    t_begin = time.time()
    output = transform3D(vol, afa, beta, (size, size, size), zoom_ratio=zoom_ratio, interp_style='linear')
    print("time1: %f" % (time.time() - t_begin))

    image = sitk.GetImageFromArray(output)
    sitk.WriteImage(image, "L:\\testFiles\\test.nii.gz")

