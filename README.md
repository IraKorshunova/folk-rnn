![My image](https://github.com/IraKorshunova/folk-rnn/blob/master/folkrnn_logo.png)

# Folk music style modelling using LSTMs

See the following websites:

3. [Let's Have Another Gan Ainm](https://soundcloud.com/oconaillfamilyandfriends) (music album)
1. [folkrnn.org](https://folkrnn.org/)
2. [The Machine Folk Session](https://themachinefolksession.org/)

Using conda, do the following:

~~~~
conda create --name folk-rnn python=2.7
conda activate folk-rnn
conda install mkl-service
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
git clone https://github.com/IraKorshunova/folk-rnn.git
cd folk-rnn
~~~~

Then to generate using one of the pretrained models:
~~~~
python sample_rnn.py --terminal metadata/folkrnn_v2.pkl
~~~~

To train a new model:
~~~~
python train_rnn.py config5 data/data_v2
~~~~

This code was used for the following published works:

1. Sturm and Ben-Tal, ["Folk the algorithms: traditional music in AI music research"](https://medium.com/the-sound-of-ai/folk-the-algorithms-traditional-music-in-ai-music-research-b19bf392d991), The Sound of AI (Medium)

1. Sturm and Ben-Tal, ["Let’s Have Another Gan Ainm: An experimental album of Irish traditional music and computer-generated tunes"](http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1248565&dswid=-8707)

1. Sturm, Ben-Tal, Monaghan, Collins, Herremans, Chew, Hadjeres, Deruty and Pachet, [“Machine learning research that matters for music creation: A case study,”](https://www.tandfonline.com/doi/full/10.1080/09298215.2018.1515233) J. New Music Research 48(1):36-55, 2018.

1. Sturm ["How Stuff Works: LSTM Model of Folk Music Transcriptions"](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxmYWltbXVzaWMyMDE4fGd4OjVjYmE2NTllMGEzMjZiM2E), invited presentation Joint Workshop on Machine Learning for Music, 2018.

1. Sturm, [“What do these 5,599,881 parameters mean? An analysis of a specific LSTM music transcription model, starting with the 70,281 parameters of its softmax layer,”](http://urn.kb.se/resolve?urn=urn%3Anbn%3Ase%3Akth%3Adiva-238604) in Proc. Music Metacreation workshop of ICCC, 2018.

1. Sturm and Ben-Tal, ["Taking the Models back to Music Practice: Evaluating Generative Transcription Models built using Deep Learning,”](https://www.jcms.org.uk/article/id/517/) J. Creative Music Systems, Vol. 2, No. 1, Sep. 2017.

1. Sturm, Santos, Ben-Tal and Korshunova, "[Music transcription modelling and composition using deep learning](https://arxiv.org/pdf/1604.08723)", in Proc. [1st Conf. Computer Simulation of Musical Creativity](https://csmc2016.wordpress.com), Huddersfield, UK, July 2016.

1. Sturm, Santos and Korshunova, ["Folk Music Style Modelling by Recurrent Neural Networks with Long Short Term Memory Units"](http://ismir2015.uma.es/LBD/LBD13.pdf), Late-breaking demo at the 2015 Int. Symposium on Music Information Retrieval

4. The folk-rnn v1, v2 and v3 Session Books https://highnoongmt.wordpress.com/2018/01/05/volumes-1-20-of-folk-rnn-v1-transcriptions/

11. 47,000+ tunes at The Endless folk-rnn Traditional Music Session http://www.eecs.qmul.ac.uk/~sturm/research/RNNIrishTrad/index.html

# Music compositions resulting from versions of this code:

1. "Cloudberry Lane" by Zoë Gorman + folk-rnn (2019) https://youtu.be/6-XDhZ_AVGQ
1. Laura Agnusdei and guest perform some output of folkrnn at the 2019 ReWire festival https://www.thoughtsource.org/vimeo_tab/app/tab/view_video?fbPageId=109823292380435&page=1&videoId=345886314
1. "Bastard Tunes" by Oded Ben-Tal + folk-rnn (v2) (2017) https://www.youtube.com/playlist?list=PLdTpPwVfxuXpQ03F398HH463SAE0vR2X8
1. "Safe Houses" by Úna Monaghan + folk-rnn (v2) (for concertina and tape, 2017) https://youtu.be/x6LS9MbQj7Y
1. "Interpretations of Computer-Generated Traditional Music" by John Hughes + folk-rnn (v2) (for double bass, 2017) https://youtu.be/GmwYtNgHW4g
1. "Dialogues with folk-rnn" by Luca Tuchet + folk-rnn (v2) (for smart mandolin, 2017) https://youtu.be/pkf3VqPieoo; at NIME 2018 https://youtu.be/VmJdLqejb-E
1. "The Fortootuise Pollo" by Bob L. Sturm + folk-rnn (v1) (2017) https://soundcloud.com/sturmen-1/the-fortootuise-pollo-1
3. "March to the Mainframe" by Bob L. Sturm + folk-rnn (v2) (2017) Performed by Ensemble x.y: https://youtu.be/TLzBcMvl15M?list=PLdTpPwVfxuXrdOyjtwfokrpzfpIlnJc5o Performed by Ensemble Volans: https://soundcloud.com/sturmen-1/march-to-the-mainframe-by-bob-l-sturm-folk-rnn-v2 Score is here: https://highnoongmt.files.wordpress.com/2017/12/twoshortpieceswithaninterlude.pdf
4. "Interlude" by Bob L. Sturm + folk-rnn (v2) (2017) Performed by Ensemble x.y: https://youtu.be/NZ08dDdYh3U?list=PLdTpPwVfxuXrdOyjtwfokrpzfpIlnJc5o Performed by Ensemble Volans: https://soundcloud.com/sturmen-1/interlude-by-bob-l-sturm-folk-rnn-v2 (synthesized version: https://soundcloud.com/sturmen-1/interlude-synthesised) Score is here: https://highnoongmt.files.wordpress.com/2017/12/twoshortpieceswithaninterlude.pdf
5. "The Humours of Time Pigeon" by Bob L. Sturm + folk-rnn (v1) (2017) Performed by Ensemble x.y: https://youtu.be/1xBisQK8-3E?list=PLdTpPwVfxuXrdOyjtwfokrpzfpIlnJc5o Performed by Ensemble Volans: https://soundcloud.com/sturmen-1/the-humours-of-time-pigeon-by-bob-l-sturm-folk-rnn-v1 (synthesized version: https://soundcloud.com/sturmen-1/the-humours-time-pigeon-synthesised) Score is here: https://highnoongmt.files.wordpress.com/2017/12/twoshortpieceswithaninterlude.pdf
1. "Chicken Bits and Bits and Bobs" by Bob L. Sturm + folk-rnn (v1) (2017, 2019) https://soundcloud.com/sturmen-1/chicken-bits-and-bits-and-bobs Score is here: https://highnoongmt.files.wordpress.com/2017/04/chicken_score.pdf
6. "The Ranston Cassock" by Bob L. Sturm + folk-rnn (v1) (2016) https://youtu.be/JZ-47IavYAU (Version for viola and tape: https://highnoongmt.wordpress.com/2017/06/18/the-ranston-cassock-take-2/)
2. Tunes by folk-rnn harmonised by DeepBach (2017)
    1. "The Glas Herry Comment" by folk-rnn (v1) + DeepBach (2017) https://youtu.be/y9xJl-ljOuA
    1. "The Drunken Pint" by folk-rnn (v1) + DeepBach (2017) https://youtu.be/xJyp7vBNVA0
    1. X:633 by folk-rnn (v2) + DeepBach (2017) https://youtu.be/BUIrbZS5eXc
    1. X:7153 by folk-rnn (v2) + DeepBach (2017) https://youtu.be/tdKCzAyynu4

5. Tunes from [folk-rnn v1 session volumes](https://highnoongmt.wordpress.com/2018/01/05/volumes-1-20-of-folk-rnn-v1-transcriptions/)
    1. "The Cunning Storm" adapted and performed by Bob L. Sturm and Carla Sturm (2018) https://highnoongmt.wordpress.com/2018/01/07/the-cunning-storm-a-folk-rnn-v1-original/
    1. "The Irish Show" adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/11/23/folk-rnn-v1-tune-the-irish-show/
    1. "Sean No Cottifall" adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/11/11/folk-rnn-v1-tune-sean-no-cottifall/
    1. "Optoly Louden" adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/07/01/optoly-louden-a-folk-rnn-original/
    1. "Bonny An Ade Nullway" adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/09/18/bobby-an-ade-nullway-a-folk-rnn-v1-tune/
    1. "The Drunken Pint" adapted and performed by Bob L. Sturm (2017) https://youtu.be/omHhyVD3PD8; performed by EECSers (2017) https://youtu.be/0gosLln8Org
    1. "The Glas Herry Comment" adapted and performed by Bob L. Sturm (2017) https://youtu.be/QZh0WSjFFDs; performed by EECSers (2017) https://youtu.be/NiUAZBLh2t0
    1. "The Mal's Copporim" adapted and performed by Bob L. Sturm (2016) https://youtu.be/YMbWwU2JdLg; performed by EECSers (2017) https://youtu.be/HOPz71Bx714
    1. "The Castle Star" adapted and performed by Bob L. Sturm (2015) https://highnoongmt.wordpress.com/2015/08/12/deep-learning-for-assisting-the-process-of-music-composition-part-2/
    1. "The Doutlace" adapted and performed by Bob L. Sturm (2015) https://highnoongmt.wordpress.com/2015/08/11/deep-learning-for-assisting-the-process-of-music-composition-part-1/

5. Tunes from the [folk-rnn v2 session volumes](https://highnoongmt.wordpress.com/2018/01/05/volumes-1-20-of-folk-rnn-v1-transcriptions/)
    1. Transcriptions 1469, 1470 & 1472 performed by Torbjorn Hultmark (2016) https://youtu.be/4kLxvJ-rXDs
    1. X:488 performed by Bob L. Sturm (2017) https://youtu.be/QWvlnOqlSes; performed by EECSers (2017) https://youtu.be/QWvlnOqlSes
    2. X:4542 adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/10/06/folk-rnn-v2-tune-4542/
    3. X:2857 adapted and performed by Bob L. Sturm and Carla Sturm (2017) https://highnoongmt.wordpress.com/2017/12/02/folk-rnn-v2-tune-2857/

5. Tunes from the [folk-rnn v3 session volumes](https://highnoongmt.wordpress.com/2018/01/05/volumes-1-20-of-folk-rnn-v1-transcriptions/)
    1. "The 2714 Polka" adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/09/10/the-2714-polka-a-folk-rnn-original/
    1. X:1166 Performed by Weltsauerstoff (2018) https://soundcloud.com/weltsauerstoff/1166-conquest-of-brittany
    1. X:1166 adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/09/16/folk-rnn-v3-tune-1166/, https://youtu.be/avxXRNJvUMk (2018)
    1. X:1650 adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/09/17/folk-rnn-v3-tune-1650/
    1. X:6197 adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/09/25/folk-rnn-v3-tune-6197/
    2. X:8589 (A Derp Deep Learning Ditty) adapted and performed by Bob L. Sturm (2017) https://highnoongmt.wordpress.com/2017/10/08/a-derp-deep-learning-ditty/

6. "Why are you and your 6 million parameters so hard to understand?" by folk-rnn (v2) and Bob L. Sturm (2018) https://highnoongmt.wordpress.com/2018/01/21/making-sense-of-the-folk-rnn-v2-model-part-8/
6. "Two Burner Brew No. 1" by folk-rnn (v2) and Bob L. Sturm and Carla Sturm (2018) https://soundcloud.com/sturmen-1/two-burner-brew-no-1
6. "A Windy Canal" by folk-rnn (v2) and Bob L. Sturm (2018) https://highnoongmt.wordpress.com/2018/01/11/making-sense-of-the-folk-rnn-v2-model-part-6/
7. "Swing Swang Swung" by folk-rnn (v2) and Bob L. Sturm (2018) https://youtu.be/Y_DBRg6SK7E (analysis here: https://highnoongmt.wordpress.com/2018/01/03/making-sense-of-the-folk-rnn-v2-model-part-5/ )
7. "Experimental lobotomy of a deep network with subsequent stimulation (2)" by folk-rnn (v2 with lobotomy) and Bob L. Sturm, Carla Sturm (2018) https://highnoongmt.wordpress.com/2018/01/13/making-sense-of-the-folk-rnn-v2-model-part-7/
6. "The Millennial Whoop Reel" by Bob L. Sturm + folk-rnn (v2) (2016) https://highnoongmt.wordpress.com/2016/08/29/millennial-whoop-with-derp-learning-a-reel/
6. "The Millennial Whoop Jig" by Bob L. Sturm + folk-rnn (v2) (2016) https://highnoongmt.wordpress.com/2016/08/28/millennial-whoop-with-derp-learning/
6. "Eight Short Outputs ..." by folk-rnn (v1) + Bob L. Sturm (2015) https://highnoongmt.wordpress.com/2015/12/20/eight-short-outputs-now-on-youtube/
6. "Carol of the Cells" by Bob L. Sturm + folk-rnn (v2) (2017) https://highnoongmt.wordpress.com/2017/12/16/carol-of-the-cells-from-the-ai-to-the-orchestra/
6. "It came out from a pretrained net" by Bob L. Sturm + folk-rnn (v2) (2016) https://highnoongmt.wordpress.com/2016/12/24/taking-a-christmas-carol-toward-the-dodecaphonic-by-derp-learning/
7. “We three layers o’ hidd’n units are” by Bob L. Sturm + folk-rnn (v2) (2015) https://highnoongmt.wordpress.com/2015/12/16/tis-the-season-for-some-deep-carols/
8. "The March of Deep Learning" by Bob L. Sturm + folk-rnn (v1) (2015) https://highnoongmt.wordpress.com/2015/08/15/deep-learning-for-assisting-the-process-of-music-composition-part-4/

# Media
1. July 2 2019, "Deep learner generated Irish folk music", Bruce Sterling [Wired](https://www.wired.com/beyond-the-beyond/2019/07/deep-learner-generated-irish-folk-music/)
2. April 15 2019 The Why Factor, BBC, ["Separating the art from the artist"](https://www.bbc.co.uk/sounds/play/w3csytzb)
3. April 17 2019 Svergies Radio, Musikguiden P3, ["Artist + AI = evigt liv"](https://sverigesradio.se/sida/avsnitt/1270356?programid=4067)
3. March 2019 Mother Jones, Clive Thompson, [What Will Happen When Machines Write Songs Just as Well as Your Favorite Musician?](https://www.motherjones.com/media/2019/03/what-will-happen-when-machines-write-songs-just-as-well-as-your-favorite-musician/)
2. February 19 2019 BBC Sounds, Science and Stuff, [How music works](https://www.bbc.co.uk/sounds/play/m0002lnh)
3. February 26 2019 BBC Click, [Bach vs. AI](https://www.bbc.co.uk/sounds/play/p0720zg8)
1. Dec. 18 2018, "How to make sweet sounding music with a hard drive", [BBC Future](http://www.bbc.com/future/story/20181217-the-musical-geniuses-that-cannot-hear)
2. Dec. 18 2018, "These AI startups are creating chart-worthy music with algorithms" [TechWorld](https://www.techworld.com/tech-innovation/these-ai-startups-are-creating-chart-worthy-music-with-algorithms-3689580/)
2. Nov. 25 2018 "Missing Link: Musik ohne Musiker? KI schwingt den Taktstock " [Heise Online](https://www.heise.de/newsticker/meldung/Missing-Link-Musik-ohne-Musiker-KI-schwingt-den-Taktstock-4224798.html)
1. Nov. 2 2018, "Så har en artificiell intelligens skapat 100.000 folkmusiklåtar", [Dagens Nyheter](https://www.dn.se/kultur-noje/sa-har-en-artificiell-intelligens-skapat-100000-folkmusiklatar/)
1. Sep. 2018, "AI created more than 100,000 pieces of music after analyzing Irish and English folk tunes", [KTH research news](https://www.kth.se/en/forskning/artiklar/ai-created-more-than-100-000-pieces-of-music-after-analyzing-irish-and-english-folk-tunes-1.845897)
2. Sep. 2018, "Do computers have musical intelligence?", [Science Node](https://sciencenode.org/feature/AI%20folk%20music.php)
3. Sep. 2018, "AI skapar 100 000 folkmusiklåtar", [Musikindustrin (Sweden)](http://www.musikindustrin.se/2018/10/23/ai-skapar-100-000-folkmusiklatar/)
1. May 2018, "No humans required? If computers can be taught to compose creatively, what does that mean for the future of music?", BBC Music Magazine, Alex Marshal
5. Feb. 28 2018, "Is music about to have its first AI No.1?" https://www.bbc.co.uk/music/articles/0c3dc8f7-4853-4379-b0d5-62175d33d557
2. Dec. 23 2017 "AI Has Been Creating Music and the Results Are...Weird" PC Mag (http://uk.pcmag.com/news/92577/ai-has-been-creating-music-and-the-results-areweird)
2. Nov. 18, 2017 Le Tube avec Stéphane Bern et Laurence Bloch, France https://www.youtube.com/watch?v=LQQER9479Xk
1. Sep. 25 2017, "Intelligenza artificiale crea più di 100.000 nuovi brani musicali folk" [Notizie scientifiche.it](https://notiziescientifiche.it/intelligenza-artificiale-crea-piu-di-100-000-nuovi-brani-musicali-folk/)
2. June 3 2017, "An A.I. in London is Writing Its Own Music and It Sounds Heavenly" https://www.inverse.com/article/32276-folk-music-ai-folk-rnn-musician-s-best-friend
2. June 8 2017, "Computer program created to write Irish trad tunes" http://www.irishtimes.com/business/technology/computer-program-created-to-write-irish-trad-tunes-1.3112238
3. June 19 2017 "Folk-RNN is the Loquantur Rhythm artist of June" (providing music for phone call waits) https://zc1.campaign-view.com/ua/SharedView?od=11287eca6b3187&cno=11a2b0b20c9c037&cd=12a539b2f47976f3&m=4 (Here is a sample: https://highnoongmt.wordpress.com/2017/06/17/deep-learning-on-hold/)
3. June 18 2017, "Real Musicians Evaluate Music Made by Artificial Intelligence" https://motherboard.vice.com/en_us/article/irish-folk-music-ai
4. June 1 2017, "Can an AI Machine Hold Copyright Protection Over Its Work?" https://artlawjournal.com/ai-machine-copyright/
2. May 26 2017 The Daily Mail named our project "Bot Dylan" (http://www.dailymail.co.uk/sciencetech/article-4544400/Researchers-create-computer-writes-folk-music.html), and then didn't even link to this page. Plus the video they edited has no computer-generated music in it. Well done!
2. April 13 2017 “Eine Maschine meistert traditionelle Folk-Music” http://www.srf.ch/kultur/netzwelt/eine-maschine-meistert-traditionelle-folk-music
1. March 31 2017, "‘Machine folk’ music composed by AI shows technology’s creative side", The Conversation (https://theconversation.com/machine-folk-music-composed-by-ai-shows-technologys-creative-side-74708)
