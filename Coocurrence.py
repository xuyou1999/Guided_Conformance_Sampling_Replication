import math


class coocurence_rule:
    def __init__(self):
        self.correlation = 0.0
        self.occurred_deviation = 0
        self.occurred_conforming = 0
        self.not_occurred_deviation = 0
        self.not_occurred_conforming = 0
        self.trace_indices = set()

    def add_conforming(self):
        self.occurred_conforming += 1

    def add_deviating(self):
        self.occurred_deviation += 1

    def add_unrelated_to_conforming(self):
        self.not_occurred_conforming += 1

    def add_unrelated_to_deviating(self):
        self.not_occurred_deviation += 1

    def add_trace_index(self, index):
        self.trace_indices.add(index)

    def add_trace_indices(self, indices):
        self.trace_indices.update(indices)

    def update_correlation(self):
        deviations = self.occurred_deviation + self.not_occurred_deviation
        conforming = self.occurred_conforming + self.not_occurred_conforming
        not_occuring = self.not_occurred_deviation + self.not_occurred_conforming
        occuring = self.occurred_conforming + self.occurred_deviation
        contingency_root = math.sqrt(deviations * conforming * not_occuring * occuring)

        #assign worse than possible correlation for cases where correlation cant be computed...
        #if contingency table is empty in one column or row, skip (correlation not measurable)
        if(contingency_root == 0.0):
            return
        # phi coefficient of two binary variables (occuring/not occuring and conformance/non-conformance)
        self.correlation = ((self.occurred_deviation * self.not_occurred_conforming) - (
                self.occurred_conforming * self.not_occurred_deviation)) / contingency_root

    def __repr__(self):
        return "OC: %s, OD:%s, NC: %s, ND: %s --> %s" % (
            self.occurred_conforming, self.occurred_deviation, self.not_occurred_conforming,
            self.not_occurred_deviation, self.correlation)
