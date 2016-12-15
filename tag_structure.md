# Structure of text base tags #

### Phoneme ###
* [0-4] **actual and nearby phonemes** *(before and after 2)*
* [5] **stress** *(1=primary, 2=secondary, 0=no stress, 3=no data)*
* [6-7] **phoneme position** *(number of phonemes before and after in the word)*
* [8-9] **distance of a primary phoneme** *(before and after)*

### Word ###
* [10] **type**
* [11-12] **position in the sentence** *(words before and after)*
* [13-15] **number of phonemes** *(in actual, before and after in the word)*

### Sentence ###
* [16] **number of words**
* [17] **number of phonemes**
