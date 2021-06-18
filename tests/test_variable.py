from fuzzy_expert.variable import FuzzyVariable


class TestVariable:
    def __init__(self):
        self.score = FuzzyVariable(
            name="score",
            universe=(150, 200),
            terms={
                "High": [(175, 0), (180, 0.2), (185, 0.7), (190, 1)],
                "Low": [(155, 1), (160, 0.8), (165, 0.5), (170, 0.2), (175, 0)],
            },
        )

    def test_interpolate(self):

        assert self.score.fuzzificate(value=170, term="High") == 0
        assert self.score.fuzzificate(value=200, term="High") == 1
        assert self.score.fuzzificate(value=187.5, term="High") == 0.85

    def test_interpolate_with_modifiers(self):

        assert (
            self.score.fuzzificate(value=187.5, term="High", modifiers=("NOT",)) == 0.15
        )

        assert (
            self.score.fuzzificate(value=187.5, term="High", modifiers=("NOT", "NOT"))
            == 0.85
        )
