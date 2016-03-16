#!/usr/bin/env python

import argparse
import math
from random import *

import midi

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description='''Script to prepare an audio dataset. Example usage:''')

parser.add_argument('midifile_in', help='Input MIDI file')
parser.add_argument('midifile_out', help='Output MIDI file')

args = parser.parse_args()
pattern = midi.read_midifile(args.midifile_in)

newpattern = midi.Pattern(format=pattern.format, resolution=pattern.resolution)
track = midi.Track()
newpattern.append(track)

for item in pattern[0]:
    if isinstance(item, midi.events.NoteOnEvent):
        flag_splitnote = randint(0, 1)
        vel = item.data[1] + (randint(0, 60) - 30)
        if randint(0, 1):
            notetorelease = 64
        else:
            notetorelease = 63
        on = midi.NoteOnEvent(tick=0, velocity=vel, pitch=notetorelease)
        track.append(on)
    elif isinstance(item, midi.events.NoteOffEvent):
        if flag_splitnote:
            numsplits = randint(1, 1)
            if numsplits == 1:
                off = midi.NoteOffEvent(tick=int(math.ceil(item.tick / 2)), pitch=notetorelease)
                track.append(off)
                vel = on.data[1] + (randint(0, 60) - 30)
                on = midi.NoteOnEvent(tick=0, velocity=vel, pitch=notetorelease)
                track.append(on)
                off = midi.NoteOffEvent(tick=int(math.ceil(item.tick / 2)), pitch=notetorelease)
            else:
                for i in range(1, numsplits):
                    off = midi.NoteOffEvent(tick=int(math.ceil(item.tick / (numsplits + 1))), pitch=notetorelease)
                    track.append(off)
                    vel = on.data[1] + (randint(0, 60) - 30)
                    on = midi.NoteOnEvent(tick=0, velocity=vel, pitch=notetorelease)
                    track.append(on)
                off = midi.NoteOffEvent(tick=int(math.ceil(item.tick / (numsplits + 1))), pitch=notetorelease)

        else:
            off = midi.NoteOffEvent(tick=item.tick, pitch=notetorelease)
        track.append(off)
    else:
        track.append(item)

# print(newpattern)
midi.write_midifile(args.midifile_out, newpattern)
