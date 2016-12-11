import text_process as tp
from scipy.io import wavfile
import h5py
import numpy as np
import pysptk
import utils as utils
import librosa

# audio file rate
rate = 16000

# frame data for pitch and mel-cepstrum
frame_size = 512
frame_step = 80

# order and alpha of mel-cepstrum
order = 25
alpha = 0.41

'''
returns the number of frames for a given time (in sec)
'''
def get_frame_count(time):
    return round(time*rate)

'''
returns the sentences as list from a given file
'''
def get_sentences(filename):
    content = []
    with open(filename) as f:
        content = f.readlines()
    for i in range(0,len(content)):
        content[i] = content[i].split("_")[2]

    return content

'''
returns phoneme and timing data for 1 sentence from a given file
'''
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

'''
calculates the time for all phonemes in 1 sentence
inputs:
    phonemes: list of phonemes from written source
    times: list of phonemes and time data from audio source
returns: 
    list of written phonemes with time data
'''
def get_phoneme_timings(phonemes, times):
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

'''
creates input and output data for the network
inputs:
    setence_source: written sentences path
    data_source_tag: begining of each sentence filename
    start_number: number of the first sentence
    end_number: number of the last sentence (!not after the last!)
outputs:
    list of training data
    network inputs:
        [0:200] kvintphone with one-hot encoding
        [200:213] phoneme tags
        [213] number of frames in the phoneme
        [214] frame position in the phoneme
    network outputs:
        [215] pitch for frame
        [216:242] mc data for frame
'''
def generate_data(senteces_source = 'data_raw/txt.done.data', data_source_tag = 'arctic_a000', start_number = 1, end_number = 2):
    data_input = []
    senteces = get_sentences(senteces_source)
    for i in  range(start_number, end_number+1):
        if "'" not in senteces[i-1] and '-' not in senteces[i-1]:
            print(i)
            phonemes = tp.generate_tags(senteces[i-1])
            times = get_timing('data_raw/lab/'+data_source_tag+str(i)+'.lab')
            sound = wavfile.read('data_raw/wav/'+data_source_tag+str(i)+'.wav')[1]
            phoneme_timings = get_phoneme_timings(phonemes=phonemes, times=times)
        
            frames = librosa.util.frame(sound, frame_length=frame_size, hop_length=frame_step).astype(np.float64).T
            frames *= pysptk.blackman(frame_size)
            pitch = pysptk.swipe(sound.astype(np.float64), fs=rate, hopsize=frame_step, min=60, max=240, otype="pitch")
            mc = np.apply_along_axis(pysptk.mcep, 1, frames, order, alpha,etype=1,eps=0.1)

            for pt in range(0,len(phoneme_timings)):
                phoneme_timing = phoneme_timings[pt]
                start_time = get_frame_count(phoneme_timing[1])
                end_time = get_frame_count(phoneme_timing[1]+phoneme_timing[2])
                if start_time%frame_step < frame_step/2:
                    start_time = start_time+frame_step - start_time%frame_step
                else:
                    start_time = start_time- start_time%frame_step

                if end_time%frame_step < frame_step/2:
                    end_time = end_time+frame_step - end_time%frame_step
                else:
                    end_time = end_time- end_time%frame_step

                t1 = round((start_time)/frame_step)
                t2 = round(end_time/frame_step)+1
                for k in range(t1,t2): 
                    data_in = []
                    for p in phonemes[phoneme_timing[0]][1]:
                        data_in.append(p)
                    data_in.append(t2)
                    data_in.append(k/(t2-t1))
                    data_in.append(pysptk.swipe(sound[k*frame_step:k*frame_step+frame_size].astype(np.float64), fs=rate, hopsize=frame_size, min=60, max=240, otype="pitch"))
                    data_in.extend(mc[k])
                    data_in = np.asarray(data_in,dtype=float)
                    data_input.append(data_in)
            
            print(len(data_input))

    return data_input

'''
creates inputs and outputs as hd5 file for the network from all avaiable data
outputs:
    normalized_data.hd5
    training.hd5
    validation.hd5
    test.hd5
'''
def get_data():
    data = generate_data(start_number=1,end_number=1)
    #data.extend(generate_data(data_source_tag='arctic_a00', start_number=10,end_number=99))
    #data.extend(generate_data(data_source_tag='arctic_a0', start_number=100,end_number=597))

    normalize_by = np.zeros((242))
    for i in range(200,212):
        normalize_by[i] = 1

    data_normalized = utils.normalize_by_column(data=data, columns=normalize_by)[0]
    generate_h5(data = data_normalized, filename='normalized_data')
    train_data, validat_data, test_data = utils.train_validate_test(data, 0.8, 0.15, 0.05)
    generate_h5(data = train_data, filename='training')
    generate_h5(data = validat_data, filename='validation')
    generate_h5(data = test_data, filename='test')

'''
creates hd5 file from a given data with a given filename
'''
def generate_h5(data, filename = 'train_set'):
    h5f = h5py.File(filename+'.h5', 'w')
    h5f.create_dataset('dataset',data=data)
    h5f.close()
    
'''
runs the script it can be remove later
'''
get_data()
