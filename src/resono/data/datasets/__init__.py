from resono.data.datasets import gaps, goat, guitarset, guitartechs

# Registry maps dataset name → module exposing download(), preprocess(), partition().
# To add a new dataset: implement those three functions following base.DatasetModule,
# add it here, and give it a subpackage under datasets/.
#
# One module can emit more than one dataset name: guitarset writes 'guitarset'
# from the microphone and 'guitarset-pickup' from the pickup, since they share
# every annotation and differ only in which recording they read.
REGISTRY = {
    "guitarset": guitarset,
    "guitartechs": guitartechs,
    "gaps": gaps,
    "goat": goat,
}
