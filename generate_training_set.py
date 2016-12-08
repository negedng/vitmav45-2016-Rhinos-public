import text_process as tp
from scipy.io import wavfile
import h5py
import numpy as np
import pysptk
import utils as utils

rate = 16000
window_size = 0.025
window_step = 0.005

window_size = round(window_size*rate)
window_step = round(window_step*rate)

def get_sound_number(time):
    return round(time*rate)

def get_file_name(tag, i):
    return tag+i

def get_sentences(filename):
    content = []
    with open(filename) as f:
        content = f.readlines()
    for i in range(0,len(content)):
        content[i] = content[i].split("_")[2]

    return content

def get_timing(filename):
    data = []
    with open(filename) as f:
        data = f.readlines()
    data_ = []
    for i in range(1,len(data)-1):
        d = data[i].split(" ")
        phenome = ""
        for k in range(0,len(d[2])-1):
            phenome = phenome + d[2][k]
        data_.append([float(d[0]),phenome])
    return data_

def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out

def get_phoneme_lengths(phonemes, times):
    times_number = 0
    data = []
    while times_number < len(times) and times[times_number][1] == 'pau':
        times_number += 1
    for phoneme_number in range(0,len(phonemes)):
        actual_data = []
        if times_number < len(times) and times[times_number][1] == phonemes[phoneme_number][0].lower():
            actual_data.append(phoneme_number)
        elif times_number+1 < len(times) and times[times_number+1][1] == phonemes[phoneme_number][0].lower():
            times_number += 1
            actual_data.append(phoneme_number)
        if (len(actual_data) > 0):
            actual_data.append(times[times_number][0])
            end_time = times[-1][0]
            if times_number+2 < len(times) and times[times_number+1][1] == 'pau':
                end_time = times[times_number+2][0]
            elif times_number+1 < len(times):
                end_time = times[times_number+1][0]
            actual_data.append(end_time-actual_data[-1])
            data.append(actual_data)
            times_number += 1
    return data

def crop_sound(delta, sound):
    delete_numbers = []
    for i in range(0,delta):
        delete_numbers.append(i)
    sound_ = np.delete(sound, delete_numbers, None)
    
    sounds = []
    delete_numbers = []
    for i in range(0,window_step):
        delete_numbers.append(i)
    sounds.append(sound_)
    for i in range(0,round(window_size/window_step-1)):
        sounds.append(np.delete(sounds[-1], delete_numbers, None))
    return sounds

def generate_f0s(sounds):
    f0s = []
    for s in sounds:
        f0s.append(pysptk.sptk.swipe(x=s, fs=rate, hopsize=window_size, otype='f0'))
    f0s = np.asarray(f0s).flatten('F')
    return f0s


def generate_data(senteces_source = 'data_raw/txt.done.data', data_source_tag = 'arctic_a000', start_number = 1, end_number = 2):
    data_input = []
    data_output = []
    senteces = get_sentences(senteces_source)
    for i in  range(start_number, end_number+1):
        if "'" not in senteces[i-1] and '-' not in senteces[i-1]:
            print(i)
            phonemes = tp.generate_tags(senteces[i-1])
            times = get_timing('data_raw/lab/'+data_source_tag+str(i)+'.lab')
            sound = wavfile.read('data_raw/wav/'+data_source_tag+str(i)+'.wav')[1]
            sound = np.array(sound, dtype=float)
            phoneme_lengths = get_phoneme_lengths(phonemes=phonemes, times=times)

            for phoneme_length in phoneme_lengths:
                start_time = get_sound_number(phoneme_length[1])
                end_time = get_sound_number(phoneme_length[1]+phoneme_length[2])
                f0s = generate_f0s(crop_sound((start_time-window_size)%window_step, sound))

                time = start_time
                while time < get_sound_number(phoneme_length[1]+phoneme_length[2]):
                    data_in = []
                    for p in phonemes[phoneme_length[0]][1]:
                        data_in.append(p)
                    data_in.append(phoneme_length[2])
                    data_in.append(((time-start_time)/window_step)/(get_sound_number(phoneme_length[2])/window_step))
                    data_in.append(f0s[time%window_step])
                    time += window_step
                    #data_in = np.asarray(data_in,dtype=float)
                    data_input.append(data_in)
    return data_input

def get_data():
    data = generate_data(start_number=1,end_number=9)
    data.extend(generate_data(data_source_tag='arctic_a00', start_number=10,end_number=99))
    #data.extend(generate_data(data_source_tag='arctic_a0', start_number=100,end_number=597))
    #data.extend(generate_data(data_source_tag='arctic_b000', start_number=1,end_number=9))
    #data.extend(generate_data(data_source_tag='arctic_b00', start_number=10,end_number=99))
    #data.extend(generate_data(data_source_tag='arctic_b0', start_number=100,end_number=541))

    normalize_by = np.zeros((216))
    for i in range(200,216):
        normalize_by[i] = 1
    data_normalized =  data #utils.normalize_by_column(data=data, columns=normalize_by)[0]
    generate_h5(data = data_normalized, filename='normalized_data')
    train_data, validat_data, test_data = utils.train_validate_test(data, 0.8, 0.15, 0.05)
    generate_h5(data = train_data, filename='training')
    generate_h5(data = validat_data, filename='validation')
    generate_h5(data = test_data, filename='test')

def generate_h5(data, filename = 'train_set'):
    h5f = h5py.File(filename+'.h5', 'w')
    for i in range(0, len(data)):
        h5f.create_dataset('dataset_'+str(i), data=data[i])
    h5f.close()
    

get_data()
