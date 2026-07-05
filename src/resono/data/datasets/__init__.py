from resono.data.datasets import guitarset

# Registry maps dataset name → module exposing download(), preprocess(), partition().
# To add a new dataset: implement those three functions following base.DatasetModule,
# add it here, and give it a subpackage under datasets/.
REGISTRY = {
    "guitarset": guitarset,
}
