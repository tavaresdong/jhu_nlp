import random

class CFGRandomSentenceGenerator(object):
    def __init__(self, filename):
        self.rules_line = self._read_grammar_file(filename)
        self.rules = self._create_rule_dict()
        # for key, value in self.rules.items():
        #     print(key)
        #     print(value)

    def _read_grammar_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [x if x.find('#') == -1 else x[:x.find('#')] for x in lines]
            lines = [x.strip() for x in lines if x.strip()]

            # For now we ignore the beginning number
            rules = [x.split()[1:] for x in lines]
        return rules

    def _create_rule_dict(self):
        rule_dict = dict()
        for rule in self.rules_line:
            lhs, rhs = rule[0], rule[1:]
            if lhs in rule_dict:
                rule_dict[lhs].append(rhs)
            else:
                rule_dict[lhs] = [rhs]
        return rule_dict

    def _contains_nonternimal(self, sentence):
        for token in sentence:
            if token in self.rules:
                return True
        return False

    def _expand(self, sentence):
        expanded_sentence = []
        for token in sentence:
            if token in self.rules:
                possible_expands = self.rules[token]
                chosen = random.choice(possible_expands)
                expanded_sentence.extend(chosen)
            else:
                expanded_sentence.append(token)
        return expanded_sentence

    def generate_sentence(self):
        sentence = ['ROOT']
        while self._contains_nonternimal(sentence):
            sentence = self._expand(sentence)
        return sentence

if __name__ == "__main__":
    generator = CFGRandomSentenceGenerator("grammar.gr")
    print(' '.join(generator.generate_sentence()))