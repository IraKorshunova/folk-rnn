# Folk music style modelling using LSTMs

This is the data we have used.

1. sessions_data_clean.txt: This comes from the [weekly repository of thesession.org](https://github.com/adactio/TheSession-data), except we have cleaned it to a very thorough degree (removing html, comments, etc.).

2. allabcwrepeats_parsed: This is the above but cleaned again (see paper), and tokenised.

3. allabcworepeats_parsed: This is the above but with the repeats made explicit (converting to midi and back to abc)

4. allabcwrepeats_parsed_wot: This is version 3 of the dataset from thesession.org. In this version, we transpose all tunes to have the root C, transpose them all to have the root C#, remove the titles, and make new mode tokens, K:maj, K:min, K:dor, and K:mix. There are over 46,000 transcriptions here.

4. midi.tgz: This is a dataset of MIDI files created from the ABC transcriptions in the file, "temp.txt". See buildset.sh for details. This comes from allabcwrepeats_parsed_wot.
