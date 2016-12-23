# vitmav45-2016-Rhinos

This is the Rhinos team deep learning project for BME vitmav45 course in 2016.
<br>
Detailed Hungarian documentation is available [here](https://github.com/BME-SmartLab-Education/vitmav45-2016-Rhinos/blob/master/doc/documentation.pdf).<br>
WAV samples are available in [doc/wav_samples](https://github.com/BME-SmartLab-Education/vitmav45-2016-Rhinos/tree/master/doc/wav_samples).

### Goal
Creating a DNN based TTS system.

###### Achieved:
* text tagging (preprocessing)
* pitch and voiced / unvoiced prediction (with working trained networks)
* mel-cepstrum spectral parameter prediction
* phoneme length prediction
* speak generation from text

###### Future:
* more training on new data for better qualty
* extends tags and add mor spectral parameters
* ui

### Results
[Results.ipynb](https://github.com/BME-SmartLab-Education/vitmav45-2016-Rhinos/blob/master/Results.ipynb) contains our results, it can be compiled standalone with the required packages (all other required files in the repository).

### Install

To generate training data with [generate_dataset.py](https://github.com/BME-SmartLab-Education/vitmav45-2016-Rhinos/blob/master/generate_dataset.py) (generate_data() method), [CMU ARCTIC](http://festvox.org/cmu_arctic/) database required in the data_raw folder in a given structure.

Or you can download our preprocessed [dataset](https://s3-us-west-2.amazonaws.com/rhinos-datasets/dataset_5.zip) and unzip it into the preprocessed_data folder.

Our trained models can be found in the models_data folder, model notebooks files can be used for training.

### Requirements
* numpy
* matplotlib
* librosa
* h5py
* pysptk
* scipy
* keras
* nltk (with corpus downloaded)

### Scripts

* [text_process.py](https://github.com/BME-SmartLab-Education/vitmav45-2016-Rhinos/blob/master/text_process.py) - creates phoneme based tags from a sentence [```generate_tags(sentence)```]
  * more from the tags in the [tag_structure.md](https://github.com/BME-SmartLab-Education/vitmav45-2016-Rhinos/blob/master/tag_structure.md)
* [generate_dataset.py](https://github.com/BME-SmartLab-Education/vitmav45-2016-Rhinos/blob/master/generate_dataset.py) - main function is to generate training, validation and test data
  * specific parameters can be set inside the script
  * ```generate_data()``` - creates h5 files and json files for the sentences
  * ```generate_sentence_data(sentence_data)``` - creates frame based tags for the given sentence (details in the comment in the file)
* [utils.py](https://github.com/BME-SmartLab-Education/vitmav45-2016-Rhinos/blob/master/utils.py) - some basic functions like standardize given columns in a matrix

### Contributors
* Németh Gergely - sound processing, network design and training
* Szántó Tamás - text processing, network design and training, AWS management

### Further contributors
* Benda Krisztián - network design research
* László Dániel - AWS setup
* Tömpe Boldizsár
