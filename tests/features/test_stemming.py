import unittest
from nala.structures.data import Dataset, Document, Part, Token
from nala.features.stemming import PorterStemFeatureGenerator


class TestPorterStemFeatureGenerator(unittest.TestCase):
    def setUp(self):
        part = Part('Make making made. Try tried tries.')
        part.sentences = [[Token('Make'), Token('making'), Token('made')],
                          [Token('try'), Token('tried'), Token('tries')]]

        self.dataset = Dataset()
        self.dataset.documents['doc_1'] = Document()
        self.dataset.documents['doc_1'].parts['part_1'] = part

        self.generator = PorterStemFeatureGenerator()

    def test_generate(self):
        self.generator.generate(self.dataset)
        features = [token.features for token in self.dataset.tokens()]
        expected = iter([{'stem[0]': 'Make'}, {'stem[0]': 'make'}, {'stem[0]': 'made'},
                         {'stem[0]': 'tri'}, {'stem[0]': 'tri'}, {'stem[0]': 'tri'}])
        for feature in features:
            self.assertEqual(feature, next(expected))


if __name__ == '__main__':
    unittest.main()