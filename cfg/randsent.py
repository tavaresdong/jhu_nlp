'''
Solution to some questions
2.(a) why does the program generate so many long sentences?
      Because we have 50% change of selecting the rule 1	NP	NP PP
      and this rule leads to repeated NPs which makes the sentence long

2.(b) The grammar allows multiple adjectives, why do your program rarely do this?
      because there are many rules for Noun, the rule Noun	Adj Noun is rarely selected

2.(d) We need to modify the probabilities for the rules mentioned in question 2.(a) and 2.(b)

2.(e) experiment with your knowledge for english grammar

2.(f) Modify generator so that it will terminate on any grammar
      Set a upperbound for #expansions, the empirical upperbound is set to 250
      If one experiment exceeds the expansion upperbound, we can re-run the expansion and
      return the result
'''
import numpy as np

class CFGRandomSentenceGenerator(object):
    MAX_EXPANSION = 250
    def __init__(self, filename):
        self.rules_line = self._read_grammar_file(filename)
        self.rules, self.rules_prob = self._create_rule_dict()

    def _read_grammar_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [x if x.find('#') == -1 else x[:x.find('#')] for x in lines]
            lines = [x.strip() for x in lines if x.strip()]

            # For now we ignore the beginning number
            rules = [x.split() for x in lines]
        return rules

    def _create_rule_dict(self):
        rule_dict = dict()
        rule_prob_dict = dict()
        for rule in self.rules_line:
            odds, lhs, rhs = float(rule[0]), rule[1], rule[2:]
            if lhs in rule_dict:
                rule_dict[lhs].append(rhs)
                rule_prob_dict[lhs].append(odds)
            else:
                rule_dict[lhs] = [rhs]
                rule_prob_dict[lhs] = [odds]
        for key in rule_prob_dict:
            odds_list = rule_prob_dict[key]
            sum_odds = sum(odds_list)
            rule_prob_dict[key] = [x / sum_odds for x in odds_list]

        # for key, value in rule_prob_dict.items():
        #     print(key)
        #     print(value)
        return rule_dict, rule_prob_dict

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
                indexes = list(range(len(possible_expands)))
                probabilities = self.rules_prob[token]
                chosen_idx = np.random.choice(indexes, p=probabilities)
                chosen = possible_expands[chosen_idx]
                expanded_sentence.extend(chosen)
            else:
                expanded_sentence.append(token)
        return expanded_sentence

    def generate_sentence(self):
        sentence = ['ROOT']
        n_calls = 0
        while self._contains_nonternimal(sentence) and n_calls < CFGRandomSentenceGenerator.MAX_EXPANSION:
            n_calls += 1
            print (sentence)
            sentence = self._expand(sentence)
        if n_calls == CFGRandomSentenceGenerator.MAX_EXPANSION:
            return self.generate_sentence()
        else:
            return sentence

if __name__ == "__main__":
    generator = CFGRandomSentenceGenerator("grammar.gr")
    print(' '.join(generator.generate_sentence()))