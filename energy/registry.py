class EnergyRegistry:
    def __init__(self):
        self.terms = []

    def add(self, term):
        self.terms.append(term)

    def compute(self, state):
        total = 0.0
        breakdown = {}
        for term in self.terms:
            e = term.compute(state)
            weighted_e = term.weight * e
            breakdown[term.name] = weighted_e.item()
            total += weighted_e
        return total, breakdown
