from resono.data.datasets import gaps, guitarset, guitarset_pickup, guitartechs

# Registry maps dataset name → module exposing download(), preprocess(), partition().
# To add a new dataset: implement those three functions following base.DatasetModule,
# add it here, and give it a subpackage under datasets/.
REGISTRY = {
    "guitarset": guitarset,
    "guitarset-pickup": guitarset_pickup,
    "guitartechs": guitartechs,
    "gaps": gaps,
}
