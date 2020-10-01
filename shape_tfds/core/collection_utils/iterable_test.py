import unittest
from shape_tfds.core.collection_utils import iterable


class IterableTest(unittest.TestCase):

    def test_first(self):
        self.assertEqual(iterable.first([1, 2, 3]), 1)
        with self.assertRaises(ValueError):
            iterable.first([])
        self.assertEqual(iterable.first((-1,)), -1)
        with self.assertRaises(TypeError):
            iterable.first(2)

    def test_single(self):
        self.assertEqual(iterable.single([2]), 2)
        with self.assertRaises(ValueError):
            iterable.single([])
        with self.assertRaises(ValueError):
            iterable.single((2, 3))
        with self.assertRaises(TypeError):
            iterable.single(2)

    def test_split(self):
        first, rest = iterable.split([2, 3, 4, 5])
        self.assertEqual(first, 2)
        self.assertEqual(list(rest), [3, 4, 5])
        with self.assertRaises(ValueError):
            iterable.split([])
        first, rest = iterable.split(['a'])
        self.assertEqual(first, 'a')
        self.assertEqual(list(rest), [])
        with self.assertRaises(TypeError):
            iterable.split(2)


if __name__ == '__main__':
    unittest.main()
