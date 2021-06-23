"""
Inference Method
===============================================================================

"""
from typing import List, Union
import numpy as np

# import matplotlib.pyplot as plt
# from fuzzy_expert.operators import get_modified_membership, probor, defuzzificate
# from fuzzy_expert.variable import FuzzyVariable

# from fuzzy_expert.plots import plot_fuzzy_input, plot_crisp_input


class DecompositionalInference:
    def __init__(
        self,
        and_operator,
        or_operator,
        implication_operator,
        composition_operator,
        production_link,
        defuzzification_operator,
    ):
        self.and_operator = and_operator
        self.or_operator = or_operator
        self.composition_operator = composition_operator
        self.production_link = production_link
        self.defuzzification_operator = defuzzification_operator
        self.implication_operator = implication_operator

    def __call__(self, variables, rules, **input_values):

        #
        # Components of a fis.
        #
        self.variables = variables
        self.rules = rules
        self.input_values: dict = input_values

        self.convert_inputs_to_facts()
        self.fuzzificate_facts()
        self.compute_modified_premise_memberships()
        self.compute_modified_consequence_memberships()
        self.compute_fuzzy_implication()
        self.compute_fuzzy_composition()

    #         self.compute_consequence_membership_aggregation()
    #         self.compute_consequence_cf_aggregation()
    #         self.build_infered_consequence()
    #         self.aggregate_production_memberships()
    #         self.aggregate_production_cf()
    #         self.defuzzificate()

    #         return self.defuzzificated_infered_membership, self.infered_cf

    def convert_inputs_to_facts(self):
        """
        Converts input values to FIS facts (fact_values, fact_cf=1.0).

        """
        self.fact_values: dict = {}
        self.fact_cf: dict = {}

        for key in self.input_values.keys():
            input_value: Union[tuple, float] = self.input_values[key]
            if isinstance(input_value, tuple):
                self.fact_values[key] = input_value[0]
                self.fact_cf[key] = input_value[1]
            else:
                self.fact_values[key] = input_value
                self.fact_cf[key] = 1.0

    def fuzzificate_crisp_fact(self, fact_name: str) -> None:
        """
        Fuzzificate a fact with a crisp value (i.e., fact: float)

        """
        fact_value = self.fact_values[fact_name]
        self.variables[fact_name].add_points_to_universe([fact_value])
        self.fact_values[fact_name] = np.array(
            [1 if u == fact_value else 0 for u in self.variables[fact_name].universe]
        )

    def fuzzificate_fuzzy_fact(self, fact_name: str) -> None:
        """
        Fuzzificate a fact specified as a membership function (i.e., fact: List[Tuple(float, float), ...])

        """
        fact_value = self.fact_values[fact_name]
        xp = [xp for xp, _ in fact_value]
        fp = [fp for _, fp in fact_value]
        self.variables[fact_name].add_points_to_universe(xp)
        self.fact_values[fact_name] = np.interp(
            x=self.variables[fact_name], xp=xp, fp=fp
        )

    def fuzzificate_facts(self):
        """
        Convert crisp facts (i.e., score = 123) to membership fuctiostions

        """
        for key in self.fact_values.keys():
            if isinstance(self.fact_values[key], (float, int)):
                self.fuzzificate_crisp_fact(fact_name=key)
            elif isinstance(self.fact_values[key], list):
                self.fuzzificate_fuzzy_fact(fact_name=key)

    def compute_modified_premise_memberships(self):

        for rule in self.rules:

            rule.modified_premise_memberships = {}

            for i_premise, premise in enumerate(rule.premises):

                if i_premise != 0:
                    premise = premise[1:]

                if len(premise) == 2:
                    fuzzyvar, term = premise
                    modifiers = None
                else:
                    fuzzyvar = premise[0]
                    term = premise[-1]
                    modifiers = premise[1:-1]

                rule.modified_premise_memberships[fuzzyvar] = self.variables[
                    fuzzyvar
                ].get_modified_membeship(term=term, modifiers=modifiers)

    def compute_modified_consequence_memberships(self):

        for rule in self.rules:

            rule.modified_consequence_memberships = {}

            for consequence in rule.consequences:

                if len(consequence) == 2:
                    fuzzyvar, term = consequence
                    modifiers = None
                else:
                    fuzzyvar = consequence[0]
                    term = consequence[-1]
                    modifiers = consequence[1:-1]

                rule.modified_consequence_memberships[fuzzyvar] = self.variables[
                    fuzzyvar
                ].get_modified_membeship(term=term, modifiers=modifiers)

    def compute_fuzzy_implication(self):

        #
        # Implication operators
        # See Kasabov, pag. 185
        #
        Ra = lambda u, v: np.minimum(1, 1 - u + v)
        Rm = lambda u, v: np.maximum(np.minimum(u, v), 1 - u)
        Rc = lambda u, v: np.minimum(u, v)
        Rb = lambda u, v: np.maximum(1 - u, v)
        Rs = lambda u, v: np.where(u <= v, 1, 0)
        Rg = lambda u, v: np.where(u <= v, 1, v)
        Rsg = lambda u, v: np.minimum(Rs(u, v), Rg(1 - u, 1 - v))
        Rgs = lambda u, v: np.minimum(Rg(u, v), Rs(1 - u, 1 - v))
        Rgg = lambda u, v: np.minimum(Rg(u, v), Rg(1 - u, 1 - v))
        Rss = lambda u, v: np.minimum(Rs(u, v), Rs(1 - u, 1 - v))

        implication_fn = {
            "Ra": Ra,
            "Rm": Rm,
            "Rc": Rc,
            "Rb": Rb,
            "Rs": Rs,
            "Rg": Rg,
            "Rsg": Rsg,
            "Rgs": Rgs,
            "Rgg": Rgg,
            "Rss": Rss,
        }[self.implication_operator]

        for rule in self.rules:

            rule.fuzzy_implications = {}

            for premise_name in rule.modified_premise_memberships.keys():

                for consequence_name in rule.modified_consequence_memberships.keys():

                    premise_membership = rule.modified_premise_memberships[premise_name]
                    consequence_membership = rule.modified_consequence_memberships[
                        consequence_name
                    ]
                    V, U = np.meshgrid(consequence_membership, premise_membership)
                    rule.fuzzy_implications[
                        (premise_name, consequence_name)
                    ] = implication_fn(U, V)

    def compute_fuzzy_composition(self):

        for rule in self.rules:

            rule.fuzzy_compositions = {}

            for premise_name in rule.modified_premise_memberships.keys():

                for consequence_name in rule.modified_consequence_memberships.keys():

                    implication = rule.fuzzy_implications[
                        (premise_name, consequence_name)
                    ]
                    fact_value = self.fact_values[premise_name]
                    n_dim = len(fact_value)
                    fact_value = fact_value.reshape((n_dim, 1))
                    fact_value = np.tile(fact_value, (1, implication.shape[1]))

                    if self.composition_operator == "max-min":
                        composition = np.minimum(fact_value, implication)

                    if self.composition_operator == "max-prod":
                        composition = fact_value * implication

                    rule.fuzzy_compositions[premise_name] = composition.max(axis=0)


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
