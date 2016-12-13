import text_process as tp
from scipy.io import wavfile
import h5py
import numpy as np
import pysptk
import utils as utils
import librosa
import json



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
returns:
    [[filename, sentence]]
'''
def get_sentences(filename):
    content = []
    ret = []
    with open(filename) as f:
        content = f.readlines()
    for i in range(0,len(content)):
        if len(content[i]) > 0:
            content[i] = content[i].split('''"''')
            if not any((c in "'-") for c in content[i][1]):
                ret.append([content[i][0].split(' ')[1],content[i][1]])
    return ret

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
creates input and output data for the network from one sentence
inputs:
    [filename, sentence]
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
        [242] number of frames in the phoneme
'''
def generate_sentence_data(sentence_data):
    print(sentence_data[0])

    data_input = []

    data_source_tag = sentence_data[0]
    sentence = sentence_data[1]
    
    phonemes = tp.generate_tags(sentence)
    times = get_timing('data_raw/lab/'+data_source_tag+'.lab')
    sound = wavfile.read('data_raw/wav/'+data_source_tag+'.wav')[1]
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
            data_in.append(np.int(t2-t1))
            data_in.append(np.float64(k)/(t2-t1))
            data_in.append(pysptk.swipe(sound[k*frame_step:k*frame_step+frame_size].astype(np.float64), fs=rate, hopsize=frame_size, min=60, max=240, otype="pitch"))
            data_in.extend(mc[k])
            data_in.append(np.int(t2-t1))
            #data_in = np.asarray(data_in,dtype=np.float64)
            data_input.append(data_in)
            
    print(len(data_input))
    return data_input

'''
generates data for the network from a given list of senteces with filenames
input:
    [[filename, sentence]]
output:
    list
'''
def generate_data(sentences):
    data = []
    for sentence in sentences:
        data.extend(generate_sentence_data(sentence))
    return data

'''
inputs:
    senteces path
    train_rate
    validate_rate
    test_rate
return:
    [train,validate,test]
output:
    training_senteces.json
    validation_sentences.josn
    test_sentences.json
'''
def get_train_validate_test_senteces(source_path='data_raw/txt.done.data', train_rate = 0.9, validate_rate = 0.05, test_rate = 0.05):
    sentences = get_sentences(source_path)
    sentences = utils.train_validate_test(data=sentences, train_rate=train_rate, validate_rate=validate_rate,test_rate=test_rate)

    create_json('training_senteces.json',sentences[0])
    create_json('validation_sentences.json',sentences[1])
    create_json('test_sentences.json',sentences[2])

    return sentences

'''
creates inputs and outputs as hd5 file for the network from all avaiable data
outputs:
    normalized_data.hd5
    training.hd5
    validation.hd5
    test.hd5
''' 
def get_data():
    sentences = get_train_validate_test_senteces()

    print('get test data...')

    test_data = generate_data(sentences[2])
    create_h5(data = test_data, filename='test')

    print('get validation data...')

    validate_data = generate_data(sentences[1])
    create_h5(data = validate_data, filename='validation')

    print('get training data...')

    train_data = generate_data(sentences[0])
    create_h5(data = train_data, filename='training')

    normalize_by = np.zeros((243))
    for i in range(200,215):
        normalize_by[i] = 1

    data_standardized = utils.normalize_by_column(data=train_data, columns=normalize_by)
    create_h5(data = data_standardized[0], filename='train-standardized')
    create_h5(data = data_standardized[2], filename='train-mean')
    create_h5(data = data_standardized[3], filename='train-std')
    
    

'''
creates hd5 file from a given data with a given filename
'''
def create_h5(data, filename = 'train_set'):
    data_ = np.array(data)
    h5f = h5py.File(filename+'.h5', 'w')
    h5f.create_dataset('dataset',data=data_)
    h5f.close()
   
'''
creates json file
inputs:
    filename
    data
    separators
'''
def create_json(filename, data, separators=('\n','\n')):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile,separators=separators)

'''
runs the script it can be remove later
'''
get_data()
