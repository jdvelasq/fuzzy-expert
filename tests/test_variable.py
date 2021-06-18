from fuzzy_expert.variable import FuzzyVariable


class TestVariable:
    def test_creation(self):

        score = FuzzyVariable(
            name="score",
            universe=(150, 200),
            terms={
                "High": [(175, 0), (180, 0.2), (185, 0.7), (190, 1)],
                "Low": [(155, 1), (160, 0.8), (165, 0.5), (170, 0.2), (175, 0)],
            },
        )
