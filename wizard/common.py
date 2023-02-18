import collections


def iterator_to_list_of_list(iterator: collections.Iterable):
    return [list(ele) for ele in iterator]
