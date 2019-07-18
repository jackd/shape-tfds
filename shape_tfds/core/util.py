from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections


def flatten_dicts(input_dict):
    out = {}
    for k, v in input_dict.items():
        if isinstance(v, dict):
            v = flatten_dicts(v)
            for k2, v2 in v.items():
                out['%s/%s' % (k, k2)] = v2
        else:
            out[k] = v
    return out


def nest_dicts(input_dict):
    out = {}
    for k, v in input_dict.items():
        keys = k.split('/')
        inner = out
        for k2 in keys[:-1]:
            inner = inner.setdefault(k2, {})
        inner[keys[-1]] = v
    return out


def lazy_chain(iterables):
    for i in iterables:
        for v in i:
            yield v
