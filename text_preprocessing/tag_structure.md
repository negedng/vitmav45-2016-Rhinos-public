# Structure of a tag #

### Phoneme ###
* [0] **stress** *(1=primary, 2=secondary, 0=no stress, 3=no data)*
* [1] **actual phoneme**
* [2-5] **nearby phonemes** *(before and after 2)*
* [6-7] **phoneme position** *(number of phonemes before and after in the word)*
* [8-9] **distance of a primary phoneme** *(before and after)*

### Word ###
* [10] **type**
* [11-12] **position in the sentence** *(words before and after)*
* [13-15] **number of phonemes** *(in actual, before and after in the word)*

### Sentence ###
* [16] **number of words**
* [17] **number of phonemes**
