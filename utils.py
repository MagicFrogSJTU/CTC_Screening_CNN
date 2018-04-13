# Created by Yizhi Chen. 20180412


import numpy as np


def crop(volume, l_index, output_len, fill_value=0):
    '''Crop a cube from the volume.
        It must be a cube. output_len is a num, not an array.
        #TODO Make it flexible
    '''
    result = np.ones((output_len, output_len, output_len), dtype=volume.dtype)*fill_value
    copy0 = np.array(l_index)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! quote!
    copy1 = np.array(copy0)+output_len
    paste0 = [0, 0, 0]
    paste1 = [output_len, output_len, output_len]

    for i in range(3):
        if copy0[i]<0:
            paste0[i] = -copy0[i]
            copy0[i] = 0
        if copy1[i]<0:
            raise IndexError
        if copy1[i]>volume.shape[i]:
            paste1[i] = output_len - (copy1[i]-volume.shape[i])
            copy1[i] = volume.shape[i]
        if copy0[i]>volume.shape[i]:
            raise IndexError
    result[paste0[0]:paste1[0],paste0[1]:paste1[1], paste0[2]:paste1[2]] = \
        volume[copy0[0]:copy1[0], copy0[1]:copy1[1], copy0[2]:copy1[2]]
    return result

def revert_crop(volume, l_index, output_len, decrop_input):
    '''Revert the crop action.
    '''
    copy0 = np.array(l_index)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! quote!
    copy1 = np.array(copy0)+output_len
    paste0 = [0, 0, 0]
    paste1 = [output_len, output_len, output_len]

    for i in range(3):
        if copy0[i]<0:
            paste0[i] = -copy0[i]
            copy0[i] = 0
        if copy1[i]<0:
            raise IndexError  # In this case, it would be of no sense.
        if copy1[i]>volume.shape[i]:
            paste1[i] = output_len - (copy1[i]-volume.shape[i])
            copy1[i] = volume.shape[i]
        if copy0[i]>volume.shape[i]:
            raise IndexError

    volume[copy0[0]:copy1[0], copy0[1]:copy1[1], copy0[2]:copy1[2]] =\
        decrop_input[paste0[0]:paste1[0],paste0[1]:paste1[1], paste0[2]:paste1[2]]


def revert_crop_MAXIMUM(volume, l_index, output_len, decrop_input):
    '''Revert the crop action. But Apply a Maximum process during the re-croping.
    '''
    copy0 = np.array(l_index)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! quote!
    copy1 = np.array(copy0)+output_len
    paste0 = [0, 0, 0]
    paste1 = [output_len, output_len, output_len]

    for i in range(3):
        if copy0[i]<0:
            paste0[i] = -copy0[i]
            copy0[i] = 0
        if copy1[i]<0:
            raise IndexError  # In this case, it would be of no sense.
        if copy1[i]>volume.shape[i]:
            paste1[i] = output_len - (copy1[i]-volume.shape[i])
            copy1[i] = volume.shape[i]
        if copy0[i]>volume.shape[i]:
            raise IndexError
    volume[copy0[0]:copy1[0], copy0[1]:copy1[1], copy0[2]:copy1[2]] = \
        np.maximum(decrop_input[paste0[0]:paste1[0],paste0[1]:paste1[1], paste0[2]:paste1[2]],
                   volume[copy0[0]:copy1[0], copy0[1]:copy1[1], copy0[2]:copy1[2]],)


