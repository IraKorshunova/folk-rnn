# Folk music style modelling using LSTMs

This is the data we have used.

1. data_v1: This comes from the [weekly repository of thesession.org](https://github.com/adactio/TheSession-data), except we have cleaned it to a very thorough degree (removing html, comments, etc.). Created folk-rnn (v1).

2. data_v2: This is the above but cleaned again (see paper), tokenised, and without titles. Created folk-rnn (v2).

3. data_v2_withtitles: data_v2 with titles of tunes

3. data_v2_worepeats: data_v2 with the repeats made explicit (converting to midi and back to abc)

4. data_v3: This is version 3 of the dataset from thesession.org. In this version, we transpose all tunes to have the root C, transpose them all to have the root C#, remove the titles, and make new mode tokens, K:maj, K:min, K:dor, and K:mix. There are over 46,000 transcriptions here. Created folk-rnn (v3).

4. midi.tgz: This is a dataset of MIDI files created from the ABC transcriptions in data_v2.
