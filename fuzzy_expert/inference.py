"""
Inference methods
===============================================================================

"""

# import numpy as np
# import matplotlib.pyplot as plt
# from fuzzy_expert.operators import get_modified_membership, probor, defuzzificate
# from fuzzy_expert.variable import FuzzyVariable

# from fuzzy_expert.plots import plot_fuzzy_input, plot_crisp_input


# class DecompositionalInference:
#     def __init__(
#         self,
#         and_operator,
#         or_operator,
#         implication_operator,
#         composition_operator,
#         production_link,
#         defuzzification_operator,
#     ):
#         self.and_operator = and_operator
#         self.or_operator = or_operator
#         self.composition_operator = composition_operator
#         self.production_link = production_link
#         self.defuzzification_operator = defuzzification_operator
#         self.implication_operator = implication_operator

#         #
#         # internal members
#         #
#         self.input_types = {}

#     def __call__(self, rules, **facts):

#         self.rules = rules
#         self.facts = facts

#         self.assign_input_cf()
#         self.fuzzificate()
#         self.compute_modified_premise_memberships()
#         self.compute_modified_consequence_membership()
#         self.compute_fuzzy_implication()
#         self.compute_fuzzy_composition()
#         self.compute_consequence_membership_aggregation()
#         self.compute_consequence_cf_aggregation()
#         self.build_infered_consequence()
#         self.aggregate_production_memberships()
#         self.aggregate_production_cf()
#         self.defuzzificate()

#         return self.defuzzificated_infered_membership, self.infered_cf

#     def assign_input_cf(self):

#         self.fact_values = {}
#         self.fact_cf = {}

#         for fact in self.facts.keys():
#             f = self.facts[fact]
#             if isinstance(f, tuple):
#                 self.fact_values[fact] = f[0]
#                 self.fact_cf[fact] = f[1]
#             else:
#                 self.fact_values[fact] = f
#                 self.fact_cf[fact] = 1.0

#     def fuzzificate(self):

#         #
#         # Transforms values to memberships
#         #

#         self.fact_types = {}
#         self.fuzzificated_fact_values = {}

#         for rule in self.rules:

#             for i_premise, premise in enumerate(rule.premises):

#                 if i_premise == 0:
#                     fuzzyvar = premise[0]
#                 else:
#                     fuzzyvar = premise[1]

#                 value = self.fact_values[fuzzyvar.name]

#                 if isinstance(value, (int, float)):
#                     self.fact_types[fuzzyvar.name] = "crisp"
#                     fuzzyvar.add_points_to_universe(value)
#                     membership = np.array(
#                         [1 if u == value else 0 for u in fuzzyvar.universe]
#                     )
#                     self.fuzzificated_fact_values[fuzzyvar.name] = membership

#                 if isinstance(value, list):
#                     self.fact_types[fuzzyvar.name] = "fuzzy"
#                     xp = [xp for xp, _ in value]
#                     fp = [fp for _, fp in value]
#                     fuzzyvar.add_points_to_universe(xp)
#                     self.fuzzificated_fact_values[fuzzyvar.name] = np.interp(
#                         x=fuzzyvar.universe, xp=xp, fp=fp
#                     )

#     def compute_modified_premise_memberships(self):

#         for rule in self.rules:

#             rule.modified_premise_memberships = {}
#             rule.universes = {}

#             for i_premise, premise in enumerate(rule.premises):

#                 if i_premise == 0:

#                     if len(premise) == 2:
#                         fuzzyvar, term = premise
#                         modifiers = None
#                     else:
#                         fuzzyvar = premise[0]
#                         term = premise[-1]
#                         modifiers = premise[1:-1]
#                 else:

#                     if len(premise) == 3:
#                         _, fuzzyvar, term = premise
#                         modifiers = None
#                     else:
#                         fuzzyvar = premise[1]
#                         term = premise[-1]
#                         modifiers = premise[2:-1]

#                 membership = fuzzyvar.terms[term]
#                 rule.modified_premise_memberships[
#                     fuzzyvar.name
#                 ] = get_modified_membership(membership, modifiers)

#                 rule.universes[fuzzyvar.name] = fuzzyvar.universe

#     def compute_modified_consequence_membership(self):

#         for rule in self.rules:

#             if len(rule.consequence) == 2:
#                 modifiers = None
#             else:
#                 modifiers = rule.consequence[1:-1]

#             term = rule.consequence[-1]
#             fuzzyvar = rule.consequence[0]
#             membership = fuzzyvar.terms[term]

#             rule.modified_consequence_membership = get_modified_membership(
#                 membership, modifiers
#             )

#     def compute_fuzzy_implication(self):

#         #
#         # Implication operators
#         # See Kasabov, pag. 185
#         #
#         Ra = lambda u, v: np.minimum(1, 1 - u + v)
#         Rm = lambda u, v: np.maximum(np.minimum(u, v), 1 - u)
#         Rc = lambda u, v: np.minimum(u, v)
#         Rb = lambda u, v: np.maximum(1 - u, v)
#         Rs = lambda u, v: np.where(u <= v, 1, 0)
#         Rg = lambda u, v: np.where(u <= v, 1, v)
#         Rsg = lambda u, v: np.minimum(Rs(u, v), Rg(1 - u, 1 - v))
#         Rgs = lambda u, v: np.minimum(Rg(u, v), Rs(1 - u, 1 - v))
#         Rgg = lambda u, v: np.minimum(Rg(u, v), Rg(1 - u, 1 - v))
#         Rss = lambda u, v: np.minimum(Rs(u, v), Rs(1 - u, 1 - v))

#         implication_fn = {
#             "Ra": Ra,
#             "Rm": Rm,
#             "Rc": Rc,
#             "Rb": Rb,
#             "Rs": Rs,
#             "Rg": Rg,
#             "Rsg": Rsg,
#             "Rgs": Rgs,
#             "Rgg": Rgg,
#             "Rss": Rss,
#         }[self.implication_operator]

#         for rule in self.rules:

#             rule.fuzzy_implications = {}

#             for name in rule.modified_premise_memberships.keys():

#                 premise_membership = rule.modified_premise_memberships[name]
#                 consequence_membership = rule.modified_consequence_membership
#                 V, U = np.meshgrid(consequence_membership, premise_membership)
#                 rule.fuzzy_implications[name] = implication_fn(U, V)

#     def compute_fuzzy_composition(self):

#         for rule in self.rules:

#             rule.fuzzy_compositions = {}

#             for name in rule.modified_premise_memberships.keys():

#                 implication = rule.fuzzy_implications[name]

#                 value = self.fuzzificated_fact_values[name]
#                 n_dim = len(value)
#                 value = value.reshape((n_dim, 1))
#                 value = np.tile(value, (1, implication.shape[1]))

#                 if self.composition_operator == "min":
#                     composition = np.minimum(value, implication)

#                 if self.composition_operator == "prod":
#                     composition = value * implication

#                 rule.fuzzy_compositions[name] = composition.max(axis=0)

#     def compute_consequence_membership_aggregation(self):

#         for rule in self.rules:

#             aggregated_membership = None

#             for premise in rule.premises:

#                 if aggregated_membership is None:
#                     aggregated_membership = rule.fuzzy_compositions[premise[0].name]
#                 else:
#                     other_membership = rule.fuzzy_compositions[premise[1].name]

#                     if premise[0] == "AND":
#                         if self.and_operator == "min":
#                             aggregated_membership = np.minimum(
#                                 aggregated_membership, other_membership
#                             )
#                         if self.and_operator == "prod":
#                             aggregated_membership = (
#                                 aggregated_membership * other_membership
#                             )

#                     if premise[0] == "OR":
#                         if self.and_operator == "max":
#                             aggregated_membership = np.maximum(
#                                 aggregated_membership, other_membership
#                             )
#                         if self.and_operator == "probor":
#                             aggregated_membership = probor(
#                                 [aggregated_membership, other_membership]
#                             )

#             rule.infered_membership = aggregated_membership

#     def compute_consequence_cf_aggregation(self):

#         for rule in self.rules:

#             aggregated_cf = None

#             for premise in rule.premises:

#                 if aggregated_cf is None:
#                     aggregated_cf = self.fact_cf[premise[0].name]
#                 else:
#                     other_cf = self.fact_cf[premise[1].name]

#                     if premise[0] == "AND":
#                         aggregated_cf = np.minimum(aggregated_cf, other_cf)

#                     if premise[0] == "OR":
#                         aggregated_cf = np.maximum(aggregated_cf, other_cf)

#             rule.infered_cf = aggregated_cf * rule.rule_cf

#     def build_infered_consequence(self):

#         self.infered_consequence = FuzzyVariable(
#             name=self.rules[0].consequence[0].name,
#             universe=self.rules[0].consequence[0].universe,
#         )

#         for i_rule, rule in enumerate(self.rules):
#             if rule.infered_cf >= rule.threshold_cf:
#                 self.infered_consequence[
#                     "Rule-{}".format(i_rule)
#                 ] = rule.infered_membership

#     def aggregate_production_memberships(self):
#         """Computes the output fuzzy set of the inference system."""

#         infered_membership = None

#         if self.production_link == "max":

#             for rule in self.rules:
#                 if infered_membership is None:
#                     infered_membership = rule.infered_membership
#                 else:
#                     infered_membership = np.maximum(
#                         infered_membership, rule.infered_membership
#                     )

#         self.infered_membership = infered_membership

#     def aggregate_production_cf(self):
#         """Computes the output fuzzy set of the inference system."""

#         infered_cf = None

#         for rule in self.rules:
#             if infered_cf is None:
#                 infered_cf = rule.infered_cf
#             else:
#                 infered_cf = np.maximum(infered_cf, rule.infered_cf)

#         self.infered_cf = infered_cf

#     def defuzzificate(self):

#         self.defuzzificated_infered_membership = defuzzificate(
#             universe=self.infered_consequence.universe,
#             membership=self.infered_membership,
#             operator=self.defuzzification_operator,
#         )

#     def plot(self, rules, **facts):
#         def get_position():
#             names = []
#             for rule in rules:
#                 for i_premise, premise in enumerate(rule.premises):
#                     if i_premise == 0:
#                         names.append(premise[0].name)
#                     else:
#                         names.append(premise[1].name)
#             names = sorted(set(names))
#             position = {name: i_name for i_name, name in enumerate(names)}
#             return position

#         # computation
#         self.__call__(rules, **facts)

#         n_rows = len(self.rules) + 1
#         position = get_position()
#         n_variables = len(position.keys())

#         for i_rule, rule in enumerate(rules):

#             #
#             # Plot premises
#             #
#             for i_premise, premise in enumerate(rule.premises):

#                 if i_premise == 0:
#                     varname = premise[0].name
#                 else:
#                     varname = premise[1].name

#                 i_col = position[varname]

#                 if i_col == 0:
#                     view_yaxis = "left"
#                 else:
#                     view_yaxis = False

#                 plt.subplot(
#                     n_rows,
#                     n_variables + 1,
#                     i_rule * (n_variables + 1) + i_col + 1,
#                 )

#                 view_xaxis = True if i_rule + 1 == len(rules) else False
#                 title = varname if i_rule == 0 else None

#                 if self.fact_types[varname] == "crisp":
#                     plot_crisp_input(
#                         value=facts[varname],
#                         universe=rule.universes[varname],
#                         membership=rule.modified_premise_memberships[varname],
#                         name=title,
#                         view_xaxis=view_xaxis,
#                         view_yaxis=view_yaxis,
#                     )
#                 else:
#                     plot_fuzzy_input(
#                         value=self.fuzzificated_fact_values[varname],
#                         universe=rule.universes[varname],
#                         membership=rule.modified_premise_memberships[varname],
#                         name=title,
#                         view_xaxis=view_xaxis,
#                         view_yaxis=view_yaxis,
#                     )

#             #
#             # Plot consesquence
#             #
#             plt.subplot(
#                 n_rows,
#                 n_variables + 1,
#                 i_rule * (n_variables + 1) + n_variables + 1,
#             )

#             plot_fuzzy_input(
#                 value=rule.infered_membership,
#                 universe=rule.consequence[0].universe,
#                 membership=rule.modified_consequence_membership,
#                 name=None,  # rule.consequence[0].name,
#                 view_xaxis=False,
#                 view_yaxis="right",
#             )

#         plt.subplot(
#             n_rows,
#             n_variables + 1,
#             n_rows * (n_variables + 1),
#         )

#         plot_crisp_input(
#             value=self.defuzzificated_infered_membership,
#             universe=self.infered_consequence.universe,
#             membership=self.infered_membership,
#             name=None,
#             view_xaxis=True,
#             view_yaxis="right",
#         )
#         plt.gca().set_xlabel(
#             "{} = {:.2f}".format(
#                 self.infered_consequence.name, self.defuzzificated_infered_membership
#             )
#         )
