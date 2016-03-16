#!/bin/bash 

abc2midi $1 -o temp2.mid
tempo=$2

# Synthesize accordion
python process.py temp2.mid temp.mid
timidity --reverb=0 -Ow -Ei22 -T $tempo temp.mid
mv temp.wav acc.wav

# Synthesize violin
python process.py temp2.mid temp.mid
timidity --reverb=0 -Ow -Ei41 -T $tempo temp.mid
mv temp.wav vio.wav

# Synthesize flute
python process.py temp2.mid temp.mid
timidity --reverb=0 -Ow -Ei74 -T $tempo temp.mid
mv temp.wav flu.wav

# Synthesize harp
python process.py temp2.mid temp.mid
timidity --reverb=0 -Ow -Ei46 -T $tempo temp.mid
mv temp.wav har.wav

# Synthesize drum
python process_drum.py temp2.mid temp.mid
timidity --reverb=25 -Ow -D 0 -T $tempo temp.mid
mv temp.wav dru.wav

sox -m -v 1.5 dru.wav -v 1 har.wav -v 1 flu.wav -v 1 acc.wav -v 1 vio.wav mix.wav reverb 50
