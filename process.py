#!/usr/bin/env python

import argparse
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
        vel = item.data[1] + (randint(0, 30) - 15)
        notetorelease = item.data[0]
        item.tick += 0 * randint(0, 5)
        on = midi.NoteOnEvent(tick=0, velocity=vel, pitch=notetorelease)
        track.append(on)
    elif isinstance(item, midi.events.NoteOffEvent):
        totaltick = item.tick - 0 * randint(0, 5)
        if randint(0, 1):
            off = midi.NoteOffEvent(tick=totaltick / 2, pitch=notetorelease)
            track.append(off)
            off = midi.NoteOffEvent(tick=totaltick / 2, pitch=notetorelease)
            track.append(off)
        else:
            off = midi.NoteOffEvent(tick=totaltick, pitch=notetorelease)
            track.append(off)
    else:
        track.append(item)

midi.write_midifile(args.midifile_out, newpattern)
