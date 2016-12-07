import text_process as tp
from scipy.io import wavfile
import h5py
import numpy as np

rate = 16000

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

def generate_data(senteces_source = 'txt.done.data', data_source_tag = 'arctic_a00', start_number = 10, end_number = 12, memory_length = 3000):
    data_input = []
    data_output = []
    senteces = get_sentences(senteces_source)
    for i in  range(start_number, end_number):
        if "'" not in senteces[i-1] and '-' not in senteces[i-1]:
            phonemes = tp.generate_tags(senteces[i-1])
            times = get_timing('lab/'+data_source_tag+str(i)+'.lab')
            bins = np.linspace(-1, 1, 256)
            sound = wavfile.read('wav/'+data_source_tag+str(i)+'.wav')[1]
            sound = normalize(sound)
            sound_out = (np.digitize(sound[1::], bins, right=False) - 1)[None, :]
            sound = np.digitize(sound[0:-1], bins, right=False) - 1
            sound = bins[sound][None, :, None]
            

            phoneme_number = 0
            for sound_number in range(0,sound.shape[1]):
                sound_phoneme_number = 0
                isNext = 0
                data_in = []
                data_out = []
                while (sound_phoneme_number < len(times) and sound_number < times[sound_phoneme_number][0]):
                    sound_phoneme_number = sound_phoneme_number + 1
                sound_phoneme_number = sound_phoneme_number - 1
                if times[sound_phoneme_number][1] != phonemes[phoneme_number][0].lower() and times[sound_phoneme_number][1] == phonemes[phoneme_number+1][0].lower():
                    phoneme_number = phoneme_number + 1
                for p in phonemes[phoneme_number][1]:
                    data_in.append(p/41)
                data_in.append(sound[0][sound_number][0])

                data_out = []
                data_out.append(sound_out[0][sound_number])
                data_out.append(isNext)
                data_input.append(data_in)
                data_output.append(data_out)

            
    return [data_input, data_output]

def generate_h5(data, filename = 'train_set'):
    h5f = h5py.File(filename+'in.h5', 'w')
    h5f.create_dataset('dataset', data=data)
    h5f.close()
    h5f = h5py.File(filename+'out.h5', 'w')
    h5f.create_dataset('dataset', data=data)
    h5f.close()

#generate_data()
