import unittest
from shape_tfds.core.collection_utils import sequence


class SequenceTest(unittest.TestCase):

    def test_dict_sequence(self):
        x = [0, 1, 2, 3]
        y = [10, 11, 12, 13]
        dict_seq = sequence.DictSequence(x=x, y=y)
        self.assertEqual(dict_seq[2], dict(x=2, y=12))
        self.assertEqual(len(dict_seq), 4)

    def test_zipped_sequence(self):
        x = [0, 1, 2, 3]
        y = [10, 11, 12, 13]
        zipped = sequence.ZippedSequence(x, y)
        self.assertEqual(zipped[2], (2, 12))
        self.assertEqual(len(zipped), 4)

    def test_mapped_sequence(self):
        base = range(5)
        mapped = sequence.MappedSequence(base, lambda x: x**2)
        self.assertEqual(list(mapped), [x**2 for x in base])
        self.assertEqual(mapped[3], 9)

    def test_map(self):
        self.assertEqual(
            list(sequence.Sequence.wrapped(range(5)).map(lambda x: x + 1)),
            list(range(1, 6)))
        self.assertEqual(
            list(sequence.Sequence.mapped(range(5), lambda x: x + 1)),
            list(range(1, 6)))

    def test_zipped(self):
        zipped = sequence.Sequence.zipped(x=[], y=[])
        self.assertIsInstance(zipped, sequence.DictSequence)
        zipped = sequence.Sequence.zipped([], [])
        self.assertIsInstance(zipped, sequence.ZippedSequence)


if __name__ == '__main__':
    unittest.main()
