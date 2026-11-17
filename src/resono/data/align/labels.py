"""Edit cached label arrays: translate a track, move single note starts.

A cached track's labels are three arrays on the frame grid, (6, frames) or
(frames,): onset (bool, True on each note's first frame), voiced (bool) and
pitch (float32 Hz, 0 where unvoiced). A note runs from its onset frame
until voicing ends or the next onset on the same string. Every edit here
keeps that structure: onset, voicing and pitch move together.
"""
import numpy as np

from resono.data.align.snapper import Note


def notes_of(onset: np.ndarray, pitch: np.ndarray) -> list[Note]:
    """The labelled notes, in time order; f0 is the first labelled pitch in
    the 6 frames from the onset (0 if there is none)."""
    onset, pitch = np.atleast_2d(onset), np.atleast_2d(pitch)
    notes = []
    for string, frame in zip(*np.nonzero(onset)):
        ahead = pitch[string, frame:frame + 6]
        ahead = ahead[ahead > 0]
        notes.append(Note(int(string), int(frame), float(ahead[0]) if len(ahead) else 0.0))
    notes.sort(key=lambda note: (note.frame, note.string))
    return notes


def _translate(array: np.ndarray, frames: int, fill) -> np.ndarray:
    out = np.full_like(array, fill)
    n = array.shape[-1]
    if frames == 0:
        out[...] = array
    elif frames < 0 and -frames < n:
        out[..., :n + frames] = array[..., -frames:]
    elif 0 < frames < n:
        out[..., frames:] = array[..., :n - frames]
    return out


def translate(
    onset: np.ndarray, voiced: np.ndarray, pitch: np.ndarray, frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move every label by `frames` (negative = earlier).

    Frames shifted in at the edge are unvoiced. A note whose onset is
    shifted out before frame 0 but whose voicing survives gets its onset on
    frame 0, so no voiced run is left without a start.
    """
    shape = np.shape(onset)
    onset, voiced, pitch = np.atleast_2d(onset), np.atleast_2d(voiced), np.atleast_2d(pitch)
    new_onset = _translate(onset, frames, False)
    new_voiced = _translate(voiced, frames, False)
    new_pitch = _translate(pitch, frames, 0)
    if frames < 0:
        cut = onset[:, :min(-frames, onset.shape[1])].any(1)
        new_onset[:, 0] |= cut & new_voiced[:, 0]
    return new_onset.reshape(shape), new_voiced.reshape(shape), new_pitch.reshape(shape)


def move_note_start(
    onset: np.ndarray, voiced: np.ndarray, pitch: np.ndarray,
    string: int, frame: int, target: int,
) -> bool:
    """Move the start of the note beginning at (string, frame) to `target`, in place.

    Earlier: the frames gained are voiced at the note's first pitch, taking
    them from whatever was sounding before (a note can't ring through a new
    pick on its string). Refused if another onset lies in between.
    Later: the frames lost become unvoiced. Refused if the note would vanish.
    Returns whether the note was moved.
    """
    onset, voiced, pitch = np.atleast_2d(onset), np.atleast_2d(voiced), np.atleast_2d(pitch)
    n = onset.shape[1]
    if target == frame or not (0 <= target < n and 0 <= frame < n) or not onset[string, frame]:
        return False
    if target < frame:
        if onset[string, target:frame].any():
            return False
        voiced[string, target:frame] = True
        pitch[string, target:frame] = pitch[string, frame]
    else:
        end = frame + 1
        while end < n and voiced[string, end] and not onset[string, end]:
            end += 1
        if target >= end:
            return False
        voiced[string, frame:target] = False
        pitch[string, frame:target] = 0
    onset[string, frame] = False
    onset[string, target] = True
    return True
