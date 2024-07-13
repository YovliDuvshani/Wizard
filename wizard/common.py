from collections import abc


def iterator_to_list_of_list(iterator: abc.Iterable):
    return [list(ele) for ele in iterator]
