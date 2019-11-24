from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from shape_tfds.core.collection_utils import mapping


class MappingTest(unittest.TestCase):

    def test_dict_mapping(self):
        x = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        y = {'a': 10, 'b': 11, 'c': 12, 'd': 13}
        dict_seq = mapping.DictMapping(x=x, y=y)
        self.assertEqual(dict_seq['c'], dict(x=2, y=12))
        self.assertEqual(len(dict_seq), 4)

    def test_zipped_mapping(self):
        x = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        y = {'a': 10, 'b': 11, 'c': 12, 'd': 13}
        zipped = mapping.ZippedMapping(x, y)
        self.assertEqual(zipped['c'], (2, 12))
        self.assertEqual(len(zipped), 4)

    def test_mapped_sequence(self):
        base = {'a': 3, 'b': 5, 'c': 10}
        mapped = mapping.MappedMapping(base, lambda x: x**2)
        self.assertEqual(dict(mapped.items()), {'a': 9, 'b': 25, 'c': 100})
        self.assertEqual(mapped['c'], 100)

    def test_map(self):
        self.assertEqual(
            dict(mapping.Mapping.wrapped({
                'a': 3
            }).map(lambda x: x + 1)), {'a': 4})
        self.assertEqual(
            dict(mapping.Mapping.mapped({'a': 3}, lambda x: x + 1)), {'a': 4})

    def test_zipped(self):
        zipped = mapping.Mapping.zipped(x={}, y={})
        self.assertIsInstance(zipped, mapping.DictMapping)
        zipped = mapping.Mapping.zipped({}, {})
        self.assertIsInstance(zipped, mapping.ZippedMapping)

    def test_flat_mapping(self):
        flat_map = mapping.FlatMapping(x={'a': 1, 'b': 2}, y={'c': 5})
        self.assertTrue(('x', 'a') in flat_map)
        self.assertEqual(flat_map['x', 'b'], 2)
        self.assertFalse('y' in flat_map)
        with self.assertRaises(KeyError):
            flat_map['y']
        self.assertEqual(len(flat_map), 3)

    def test_nested_flat_mapping(self):
        p = mapping.FlatMapping(x={'a': 1, 'b': 2}, y={'c': 5})
        q = mapping.FlatMapping(s={'foo': 1}, t={})
        flat_map = mapping.NestedFlatMapping(p=p, q=q)
        self.assertTrue(('p', 'x', 'a') in flat_map)
        self.assertFalse(('p', 'x') in flat_map)
        self.assertFalse(('x', 'a') in flat_map)
        with self.assertRaises(KeyError):
            flat_map['p']
            flat_map['p', 'x']
        self.assertEqual(flat_map['p', 'x', 'b'], 2)
        self.assertEqual(len(flat_map), 4)


if __name__ == '__main__':
    unittest.main()
