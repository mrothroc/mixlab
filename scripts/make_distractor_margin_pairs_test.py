import re
import unittest

from make_distractor_margin_pairs import find_structural_candidate, make_pairs


class FakeEncoding:
    def __init__(self, text):
        matches = list(re.finditer(r"[A-Za-z]+|[^\w\s]", text))
        self.offsets = [match.span() for match in matches]
        self.ids = [sum(ord(char) for char in match.group().lower()) for match in matches]


class FakeTokenizer:
    def encode(self, text):
        return FakeEncoding(text)


class MakeDistractorMarginPairsTest(unittest.TestCase):
    def test_relational_noun_pair_keeps_distractor_target_ids(self):
        words = "The key of the cabinets is rusty.".split()
        candidate = find_structural_candidate(words)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate[0], "relational_noun")
        pairs, report = make_pairs(FakeTokenizer(), ["The key of the cabinets is rusty."], 1, 7, set())
        self.assertEqual(report["written"], 1)
        pair = pairs[0]
        self.assertEqual(pair["family"], "relational_noun")
        self.assertEqual(
            [pair["view_pos"][index] for index in pair["target_pos_positions"]],
            pair["target_ids"],
        )
        self.assertEqual(
            [pair["view_neg"][index] for index in pair["target_neg_positions"]],
            pair["target_ids"],
        )

    def test_contamination_guard_rejects_source_sentence(self):
        text = "The key of the cabinets is rusty."
        pairs, report = make_pairs(FakeTokenizer(), [text], 1, 7, {"the key of the cabinets is rusty."})
        self.assertEqual(pairs, [])
        self.assertEqual(report["rejected"].get("contamination_guard"), 1)


if __name__ == "__main__":
    unittest.main()
