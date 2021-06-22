# from fuzzy_expert.variable import FuzzyVariable
# from fuzzy_expert.rule import FuzzyRule
# from fuzzy_expert.inference import DecompositionalInference


# import matplotlib.pyplot as plt

# from fuzzy_expert.mf import gaussmf

# print(gaussmf("x", 1, 1))

# import os
# os.chdir('/workspaces/fuzzy-expert/')
# !pwd


# rule_1 = FuzzyRule(
#     cf=0.8,
#     premises=[
#         (score, "High"),
#         ("AND", ratio, "Goodr"),
#         ("AND", credit, "Goodc"),
#     ],
#     consequence=(decision, "Approve"),
# )


# print(rule_1)


# rule_2 = FuzzyRule(
#     cf=0.7,
#     premises=[
#         (score, "Low"),
#         ("AND", ratio, "Badr"),
#         ("OR", credit, "Badc"),
#     ],
#     consequence=(decision, "Reject"),
# )

# print(rule_2)


# model = DecompositionalInference(
#     and_operator="min",
#     or_operator="max",
#     implication_operator="Rc",
#     composition_operator="min",  # min / prod
#     production_link="max",
#     defuzzification_operator="cog",
# )


# print(
#     model(
#         rules=[rule_1, rule_2],
#         score=(190, 0.9),
#         ratio=(0.39, 1),
#         credit=(1.5, 1),
#     )
# )


# plt.figure(figsize=(12, 9))
# model.plot(
#     rules=[rule_1, rule_2],
#     score=190,
#     ratio=0.38,
#     credit=1.5,
# )


# model = DecompositionalInference(
#     and_operator="min",
#     or_operator="max",
#     implication_operator="Rc",
#     composition_operator="min",  # min / prod
#     production_link="max",
#     defuzzification_operator="cog",
# )


# score_1 = [(180, 0.0), (190, 0.2), (195, 0.8), (200, 1.0)]
# ratio_1 = [(0.1, 1), (0.3, 1), (0.4, 0.6), (0.41, 0.2), (0.42, 0)]
# credit_1 = [(0, 0), (1, 1), (2, 1), (3, 0.7), (4, 0.3), (5, 0)]

# score_2 = [(150, 0.9), (155, 0.7), (160, 0.5), (165, 0.3), (170, 0.0)]
# ratio_2 = [(0.44, 0), (0.45, 0.3), (0.5, 0.5), (0.7, 0.7), (1, 0.9)]
# credit_2 = [(6, 0), (7, 0.3), (8, 0.5), (9, 0.7), (10, 0.9)]

# score_3 = [(185, 0.0), (190, 0.4), (195, 0.6), (200, 0.8)]
# ratio_3 = [(0.45, 0), (0.5, 0.4), (0.7, 0.6), (1, 0.8)]
# credit_3 = [(2, 1), (3, 0.8), (4, 0.6), (5, 0.4), (6, 0)]


# print(
#     model(
#         rules=[rule_1, rule_2],
#         score=score_1,
#         ratio=ratio_1,
#         credit=credit_1,
#     )
# )

# plt.figure(figsize=(12, 9))
# model.plot(
#     rules=[rule_1, rule_2],
#     score=score_1,
#     ratio=ratio_1,
#     credit=credit_1,
# )
