import abc
from nalaf.structures.data import Edge
from itertools import product, chain


class EdgeGenerator(object):
    """
    Abstract class for generating edges between two entities. Each edge represents
    a possible relationship between the two entities
    Subclasses that inherit this class should:
    * Be named [Name]EdgeGenerator
    * Implement the abstract method generate
    * Append new items to the list field "edges" of each Part in the dataset
    """

    def __init__(self, entity1_class, entity2_class, relation_type):
        self.entity1_class = entity1_class
        self.entity2_class = entity2_class
        self.relation_type = relation_type


    @abc.abstractmethod
    def generate(self, dataset):
        """
        :type dataset: nalaf.structures.data.Dataset
        """
        return


class SentenceDistanceEdgeGenerator(EdgeGenerator):
    """
    Simple implementation of generating edges between the two entities
    if they are #`distance` sentences away (always same part).

    As a special case, if `distance` is None, the distance is disregarded,
    consequently creating edges for all entities within a part (paragraph or what not).
    """

    def __init__(self, entity1_class, entity2_class, relation_type, distance, use_gold=True, use_pred=False, rewrite_edges=True):
        # Note: we could also, for example, filter edges/sentences by a list of allowed words

        super().__init__(entity1_class, entity2_class, relation_type)
        self.distance = distance

        self.use_gold = use_gold
        """Whether to generate the dataset's edges with gold annotations"""

        self.use_pred = use_pred
        """Whether to generate the dataset's edges with (pred)icted annotations"""

        self.rewrite_edges = rewrite_edges

        def chain_entities(part):
            return chain(part.annotations if self.use_gold else [], part.predicted_annotations if self.use_pred else [])

        self.part_entities = chain_entities


    def generate(self, dataset):
        import numpy

        wmin = 999
        wmax = 0

        totalcount = 0

        counts = numpy.array([0] * 104)

        for part in dataset.parts():
            if self.rewrite_edges:
                part.edges = []
            part.percolate_tokens_to_entities()

            e1_seq = (e for e in self.part_entities(part) if e.class_id == self.entity1_class)
            e2_seq = (e for e in self.part_entities(part) if e.class_id == self.entity2_class)

            for e_1, e_2 in product(e1_seq, e2_seq):
                s1_index = part.get_sentence_index_for_annotation(e_1)
                s2_index = part.get_sentence_index_for_annotation(e_2)

                if s2_index < s1_index:
                    s1_index, s2_index = s2_index, s1_index

                if e_2.offset < e_1.offset:
                    e_1, e_2 = e_2, e_1

                pair_distance = s2_index - s1_index
                assert pair_distance >= 0  # Because they must be sorted

                if pair_distance == self.distance or self.distance is None:
                    # e_2_index = e_2.tokens[0].features['id']
                    # e_1_index = e_1.tokens[-1].features['id']

                    e_2_index_alt = part.get_token_index_within_sentence_for_entity(e_2.sentence, e_2)
                    # assert(e_2_index == e_2_index_alt)
                    e_1_index_alt = part.get_token_index_within_sentence_for_entity(e_2.sentence, e_1) + (len(e_1.tokens) - 1)
                    # assert(e_1_index == e_1_index_alt)

                    num_words_distance = e_2_index_alt - e_1_index_alt
                    assert(num_words_distance >= 1)

                    totalcount += 1

                    threshold_index = num_words_distance - 1

                    counts[threshold_index:] += 1

                    # if (num_words_distance in {104, 1}):
                    #     print(num_words_distance, e_1.sentence, e_1.text, e_2.text)

                    if (num_words_distance < wmin):
                        wmin = num_words_distance
                    if (num_words_distance > wmax):
                        wmax = num_words_distance

                    if num_words_distance <= (6 * 10) + 1:
                        edge = Edge(self.relation_type, e_1, e_2, part, part, s1_index, s2_index)
                        part.edges.append(edge)

        print("***", wmin, wmax)


class CombinatorEdgeGenerator(EdgeGenerator):
    """
    Combines 2 or more edge generators.

    Assumes (does not test) that all generators have same entities and relation types properties.
    """

    def __init__(self, generator_1, generator_2, *rest):
        super().__init__(generator_1.entity1_class, generator_1.entity2_class, generator_1.relation_type)
        self.generators = [generator_1, generator_2, *rest]


    def generate(self, dataset):
        for g in self.generators:
            g.generate(dataset)
