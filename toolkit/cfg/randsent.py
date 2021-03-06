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

3. This problem mainly focused on designing Context Free Grammars, see grammar_extended for some examples
'''
import numpy as np
import sys

class SyntaxTreeNode(object):
    def __init__(self, symbol, rules):
        self.sym = symbol
        self.children = None
        if symbol in rules:
            self.is_terminal = False
        else:
            self.is_terminal = True

    def get_sentence(self):
        if self.is_terminal:
            return self.sym
        else:
            child_syms = [child.get_sentence() for child in self.children]
            return ' '.join(child_syms)

    def get_structured_sentence(self):
        if self.children is None:
            return self.sym
        else:
            child_sents = [child.get_structured_sentence() for child in self.children]
            child_sents.insert(0, self.sym)
            return '(' + ' '.join(child_sents) + ')'

    def get_higlighted_sentence(self):
        if self.children is None:
            return self.sym
        else:
            child_sents = [child.get_higlighted_sentence() for child in self.children]
            sentence  = ' '.join(child_sents)
            if self.sym == 'S':
                return '{' + sentence + '}'
            elif self.sym == 'NP':
                return '[' + sentence + ']'
            else:
                return sentence


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

        return rule_dict, rule_prob_dict

    def _contains_nonternimal(self, sentence):
        for token in sentence:
            if token in self.rules:
                return True
        return False

    def _expand_iterative(self, sentence):
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

    def generate_sentence_iterative(self):
        sentence = ['ROOT']
        n_calls = 0
        while self._contains_nonternimal(sentence) and n_calls < CFGRandomSentenceGenerator.MAX_EXPANSION:
            n_calls += 1
            sentence = self._expand_iterative(sentence)
        if n_calls == CFGRandomSentenceGenerator.MAX_EXPANSION:
            return self.generate_sentence_iterative()
        else:
            return sentence

    def _expand_tree(self, node):
        self.n_expand += 1
        if self.n_expand >= CFGRandomSentenceGenerator.MAX_EXPANSION:
            child = SyntaxTreeNode('...', self.rules)
            node.children = [child]
            return node

        if not node.is_terminal:
            token = node.sym
            possible_expands = self.rules[token]
            indexes = list(range(len(possible_expands)))
            probs = self.rules_prob[token]
            chosen_idx = np.random.choice(indexes, p=probs)
            expanded = possible_expands[chosen_idx]
            children = [SyntaxTreeNode(t, self.rules) for t in expanded]
            recursive_expanded = [self._expand_tree(child) for child in children]
            node.children = recursive_expanded

        return node

    def generate_sentence_tree(self):
        self.n_expand = 0
        root = SyntaxTreeNode('ROOT', self.rules)
        tree = self._expand_tree(root)
        return tree.get_higlighted_sentence()

def print_usage():
    print("run.py grammar_file num_sentences")
    print("e.g. randsent.py grammar.gr 5 # will generate 5 sentences according to grammar.gr")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
    else:
        grammar_file, n_sentences = sys.argv[1], int(sys.argv[2])
        generator = CFGRandomSentenceGenerator(grammar_file)
        # for i in range(n_sentences):
        #     print(' '.join(generator.generate_sentence()))
        print(generator.generate_sentence_tree())