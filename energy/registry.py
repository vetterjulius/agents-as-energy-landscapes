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
            breakdown[term.name] = e.item()
            total += term.weight * e
        return total, breakdown
