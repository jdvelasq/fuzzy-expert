"""
Inference Method
===============================================================================

"""
from __future__ import annotations

from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, widgets

from fuzzy_expert.operators import (
    bounded_prod,
    bounded_sum,
    defuzzificate,
    drastic_prod,
    drastic_sum,
    maximum,
    minimum,
    prob_or,
    product,
)
from fuzzy_expert.plots import plot_crisp_input, plot_fuzzy_input
from fuzzy_expert.variable import FuzzyVariable

# from fuzzy_expert.operators import get_modified_membership, probor, defuzzificate
#


class DecompositionalInference:
    """
    Decompositional inference method.


    :param and_operator: AND operator method for combining the compositions of propositions in a fuzzy rule premise, specified as one of the following:

        * `"min"`.
        * `"prod"`.
        * `"bunded_prod"`.
        * `"drastic_prod"`.

    :param or_operator: OR operator method for combining the compositions of propositions in a fuzzy rule premise, specified as one of the following:

        * `"max"`.
        * `"prob_or"`.
        * `"bounded_sum"`.
        * `"drastic_sum"`.

    :param implication_operator: method for computing the compositions of propositions in a fuzzy rule premise, specified as one of the following:

        * `"Ra"`.
        * `"Rm"`.
        * `"Rc"`.
        * `"Rb"`.
        * `"Rs"`.
        * `"Rg"`.
        * `"Rsg"`.
        * `"Rgs"`.
        * `"Rgg"`.
        * `"Rss"`.

    :param production_link: method for aggregating the consequences of the fuzzy rules, specified as one of the following:

        *  `"min"`.
        * `"prod"`.
        * `"bunded_prod"`.
        * `"drastic_prod"`.
        * `"max"`.
        * `"prob_or"`.
        * `"bounded_sum"`.
        * `"drastic_sum"`.

    :param defuzzification_operator: Method for defuzzificate the resulting membership function, specified as one of the following:

        * `"cog"`: Center of gravity.
        * `"boa"`: Bisector of area.
        * `"mom"`: Mean of the values for which the membership function is maximum.
        * `"lom"`: Largest value for which the membership function is maximum.
        * `"som"`: Smallest value for which the membership function is minimum.

    """

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

        self._convert_inputs_to_facts()
        self._fuzzificate_facts()
        self._compute_modified_premise_memberships()
        self._compute_modified_consequence_memberships()
        self._compute_fuzzy_implication()
        self._compute_fuzzy_composition()
        self._combine_antecedents()
        self._compute_rule_infered_cf()
        self._collect_rule_memberships()
        self._aggregate_collected_memberships()
        self._aggregate_production_cf()
        self._defuzzificate()

        return self.defuzzificated_infered_memberships, self.infered_cf

    def _convert_inputs_to_facts(self):
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

    def _fuzzificate_crisp_fact(self, fact_name: str) -> None:
        """
        Fuzzificate a fact with a crisp value (i.e., fact: float)

        """
        fact_value = self.fact_values[fact_name]
        self.variables[fact_name].add_points_to_universe([fact_value])
        self.fact_values[fact_name] = np.array(
            [1 if u == fact_value else 0 for u in self.variables[fact_name].universe]
        )

    def _fuzzificate_fuzzy_fact(self, fact_name: str) -> None:
        """
        Fuzzificate a fact specified as a membership function (i.e., fact: List[Tuple(float, float), ...])

        """
        fact_value = self.fact_values[fact_name]
        xp = [xp for xp, _ in fact_value]
        fp = [fp for _, fp in fact_value]
        self.variables[fact_name]._add_points_to_universe(xp)
        self.fact_values[fact_name] = np.interp(
            x=self.variables[fact_name].universe, xp=xp, fp=fp
        )

    def _fuzzificate_facts(self):
        """
        Convert crisp facts (i.e., score = 123) to membership fuctiostions

        """
        self.fact_types = {}
        for key in self.fact_values.keys():
            if isinstance(self.fact_values[key], (float, int)):
                self._fuzzificate_crisp_fact(fact_name=key)
                self.fact_types[key] = "crisp"
            elif isinstance(self.fact_values[key], list):
                self._fuzzificate_fuzzy_fact(fact_name=key)
                self.fact_types[key] = "fuzzy"

    def _compute_modified_premise_memberships(self):

        for rule in self.rules:

            rule.modified_premise_memberships = {}

            for i_proposition, proposition in enumerate(rule.premise):

                if i_proposition != 0:
                    proposition = proposition[1:]

                if len(proposition) == 2:
                    fuzzyvar, term = proposition
                    modifiers = None
                else:
                    fuzzyvar = proposition[0]
                    term = proposition[-1]
                    modifiers = proposition[1:-1]

                rule.modified_premise_memberships[fuzzyvar] = self.variables[
                    fuzzyvar
                ].get_modified_membeship(term=term, modifiers=modifiers)

    def _compute_modified_consequence_memberships(self):

        for rule in self.rules:

            rule.modified_consequence_memberships = {}

            for premise in rule.consequence:

                if len(premise) == 2:
                    fuzzyvar, term = premise
                    modifiers = None
                else:
                    fuzzyvar = premise[0]
                    term = premise[-1]
                    modifiers = premise[1:-1]

                rule.modified_consequence_memberships[fuzzyvar] = self.variables[
                    fuzzyvar
                ].get_modified_membeship(term=term, modifiers=modifiers)

    def _compute_fuzzy_implication(self):

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

    def _compute_fuzzy_composition(self):

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

                    rule.fuzzy_compositions[
                        (premise_name, consequence_name)
                    ] = composition.max(axis=0)

    def _combine_antecedents(self):

        for rule in self.rules:

            rule.combined_composition = {}

            for consequence_name in rule.modified_consequence_memberships.keys():

                combined_composition = None

                for proposition in rule.premise:

                    if combined_composition is None:
                        combined_composition = rule.fuzzy_compositions[
                            (proposition[0], consequence_name)
                        ]
                    else:
                        other_composition = rule.fuzzy_compositions[
                            (proposition[1], consequence_name)
                        ]

                        operator = proposition[0]

                        if operator == "AND":
                            operator = self.and_operator

                        if operator == "OR":
                            operator = self.or_operator

                        operator_fn = {
                            "min": minimum,
                            "prod": product,
                            "bunded_prod": bounded_prod,
                            "drastic_prod": drastic_prod,
                            "max": maximum,
                            "prob_or": prob_or,
                            "bounded_sum": bounded_sum,
                            "drastic_sum": drastic_sum,
                        }[operator]

                        combined_composition = operator_fn(
                            [combined_composition, other_composition]
                        )

                rule.combined_composition[consequence_name] = combined_composition

    def _compute_rule_infered_cf(self):

        for rule in self.rules:

            aggregated_premise_cf = None

            for proposition in rule.premise:

                if aggregated_premise_cf is None:
                    aggregated_premise_cf = self.fact_cf[proposition[0]]
                else:
                    other_premise_cf = self.fact_cf[proposition[1]]

                    if proposition[0] == "AND":
                        aggregated_premise_cf = np.minimum(
                            aggregated_premise_cf, other_premise_cf
                        )

                    if proposition[0] == "OR":
                        aggregated_premise_cf = np.maximum(
                            aggregated_premise_cf, other_premise_cf
                        )

            rule.infered_cf = aggregated_premise_cf * rule.rule_cf

    def _collect_rule_memberships(self):

        self.collected_rule_memberships = {}

        for i_rule, rule in enumerate(self.rules):

            for key in rule.combined_composition.keys():

                if key not in self.collected_rule_memberships.keys():

                    self.collected_rule_memberships[key] = FuzzyVariable(
                        universe_range=(
                            min(self.variables[key].universe),
                            max(self.variables[key].universe),
                        )
                    )
                    self.collected_rule_memberships[key].universe = self.variables[
                        key
                    ].universe

                if rule.infered_cf >= rule.threshold_cf:

                    self.collected_rule_memberships[key].terms[
                        "Rule-{}".format(i_rule)
                    ] = rule.combined_composition[key]

    def _aggregate_collected_memberships(self):
        """Computes the output fuzzy set of the inference system."""

        operator_fn = {
            "min": minimum,
            "prod": product,
            "bunded_prod": bounded_prod,
            "drastic_prod": drastic_prod,
            "max": maximum,
            "prob_or": prob_or,
            "bounded_sum": bounded_sum,
            "drastic_sum": drastic_sum,
        }[self.production_link]

        aggregated_memberships = {}

        for key in self.collected_rule_memberships.keys():

            fuzzyvar = self.collected_rule_memberships[key]
            memberships = [fuzzyvar.terms[term] for term in fuzzyvar.terms.keys()]

            aggregated_memberships[key] = operator_fn(memberships)

        self.aggregated_memberships = aggregated_memberships

    def _aggregate_production_cf(self):
        """Computes the output fuzzy set of the inference system."""

        infered_cf = None

        for rule in self.rules:
            if infered_cf is None:
                infered_cf = rule.infered_cf
            else:
                infered_cf = np.maximum(infered_cf, rule.infered_cf)

        self.infered_cf = infered_cf

    def _defuzzificate(self):

        self.defuzzificated_infered_memberships = {}

        for key in self.aggregated_memberships.keys():

            self.defuzzificated_infered_memberships[key] = defuzzificate(
                universe=self.variables[key].universe,
                membership=self.aggregated_memberships[key],
                operator=self.defuzzification_operator,
            )

    def plot(self, variables, rules, **facts):
        def get_position():
            position = {name: i_name for i_name, name in enumerate(variables.keys())}
            return position

        # computation
        self.__call__(variables, rules, **facts)

        n_rows = len(self.rules) + 1
        n_variables = len(variables)
        position = get_position()

        for i_rule, rule in enumerate(rules):

            #
            # Plot premises
            #
            for i_proposition, proposition in enumerate(rule.premise):

                if i_proposition == 0:
                    varname = proposition[0]
                else:
                    varname = proposition[1]

                i_col = position[varname]

                if i_col == 0:
                    view_yaxis = "left"
                else:
                    view_yaxis = False

                plt.subplot(
                    n_rows,
                    n_variables,
                    i_rule * n_variables + i_col + 1,
                )

                view_xaxis = True if i_rule + 1 == len(rules) else False
                title = varname if i_rule == 0 else None

                if self.fact_types[varname] == "crisp":
                    plot_crisp_input(
                        value=facts[varname],
                        universe=variables[varname].universe,
                        membership=rule.modified_premise_memberships[varname],
                        name=title,
                        view_xaxis=view_xaxis,
                        view_yaxis=view_yaxis,
                    )
                else:
                    plot_fuzzy_input(
                        value=self.fact_values[varname],
                        universe=variables[varname].universe,
                        membership=rule.modified_premise_memberships[varname],
                        name=title,
                        view_xaxis=view_xaxis,
                        view_yaxis=view_yaxis,
                    )

            #
            # Plot consesquence
            #
            for i_proposition, proposition in enumerate(rule.consequence):

                varname = proposition[0]
                i_col = position[varname]

                if i_col + 1 == len(variables):
                    view_yaxis = "right"
                else:
                    view_yaxis = False

                plt.subplot(
                    n_rows,
                    n_variables,
                    i_rule * n_variables + i_col + 1,
                )

                plot_fuzzy_input(
                    value=rule.combined_composition[varname],
                    universe=variables[varname].universe,
                    membership=rule.modified_consequence_memberships[varname],
                    name=None,  # rule.consequence[0].name,
                    view_xaxis=False,
                    view_yaxis="right",
                )

        for key in self.defuzzificated_infered_memberships.keys():

            varname = key
            i_col = position[varname]

            plt.subplot(
                n_rows,
                n_variables,
                (n_rows - 1) * n_variables + i_col + 1,
            )

            plot_crisp_input(
                value=self.defuzzificated_infered_memberships[key],
                universe=variables[varname].universe,
                membership=self.aggregated_memberships[key],
                name=None,
                view_xaxis=True,
                view_yaxis="right",
            )

            plt.gca().set_xlabel(
                "{} = {:.2f}".format(
                    key,
                    self.defuzzificated_infered_memberships[key],
                )
            )
