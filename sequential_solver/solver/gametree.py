import re

class Game:
    # chance player denoted as -1
    def __init__(self):
        self.players = 0
        self.root = Node()
        self.actions = []
        self.nodes = []
        self.terminals = []
        self.infosets = []

        self.parameters = []
        self.equations = []

        self.variables = []
        self.variable_map = {}
        self.variable_names = {}

        self.solutions = []

    def terminals(self):
        return self.root.terminals

    def process(self):
        self.calc_variables()
        self.calc_terminals()
        self.calc_paths()
        self.calc_names()

    def print(self):
        s = ""
        for infoset in self.infosets:
            entry = ""
            if len(infoset.nodes) == 1:
                entry += "Node " + infoset.nodes[0].name + " of Player " + str(infoset.player) +"\n"
            else:
                entry += "Information Set " + infoset.name + " of Player " + str(infoset.player) + "\n"
                node_names = []
                for node in infoset.nodes:
                    node_names.append(node.name)
                entry += "Has Nodes: " + ", ".join(node_names) + "\n"

            action_names = []
            for action in infoset.actions:
                action_names.append(action.name)
            entry += "With Actions: " + ", ".join(action_names)+ "\n"
            s += entry +"\n"
        s += "Terminal Nodes: \n"
        for tnode in self.terminals:
            payoffs = []
            for p in tnode.outcome:
                payoffs.append(str(p))
            s += tnode.name + " with payoff " + " ".join(payoffs) + "\n"

        return s

    def print_solutions(self, long=False, include_types=[]):
        solution_types = {}
        for solution in self.solutions:
            s = ""
            if long:
                for infoset in self.infosets:
                    if infoset.is_chance():
                        print("chance")
                        continue
                    s += "At " + infoset.name + ", player " + str(infoset.player) + " "
                    if len(infoset.nodes) > 1:
                        beliefs = []
                        hasbelief = True
                        for n in infoset.nodes:
                            if self.variable_map[n] not in solution.variable_constraints:
                                hasbelief = False
                                break
                            beliefs.append(solution.variable_constraints[self.variable_map[n]])
                        if hasbelief:
                            s += "believes: " + ", ".join(beliefs) + ", and "
                    actions = []
                    for a in infoset.actions:
                        actions.append(solution.variable_constraints[self.variable_map[a]])
                    s += "plays: " + ", ".join(actions) + "\n"
                s += "This results in a payoff of: \n"
                for i in range(self.players):
                    s += solution.utility[i] + " for player " + str(i) + "\n"
            else:
                constraints = []
                for var in self.variables:
                    if var in solution.variable_constraints:
                        constraints.append(solution.variable_constraints[var])
                s += "profile: " + ", ".join(constraints) + "\n"
                payoffs = []
                for i in range(self.players):
                    payoffs.append(solution.utility[i])
                s += "payoffs: " + ", ".join(payoffs) + "\n"

            if solution.type in solution_types:
                solution_types[solution.type].append(s)
            else:
                solution_types[solution.type] = [s]
        p = ""
        for type in solution_types:
            if include_types and type not in include_types:
                continue
            p += type + "\n"
            i = 1
            for s in solution_types[type]:
                p += str(i) + ":\n"
                p += s + "\n"
                i = i + 1
            p += "\n"

        any_infoset = True
        info_str = "Informationsets:\n"
        for infoset in self.infosets:
            if len(infoset.nodes) < 2:
                continue
            any_infoset = True
            lst = []
            for node in infoset.nodes:
                lst.append(node.name)
            info_str += infoset.name + ": " + ", ".join(lst) + "\n"
        if any_infoset:
            p += info_str
        pattern = re.compile(r'I\d+[AN]\d+[bp]')
        readable = pattern.sub(lambda match: self.variable_names.get(match.group(0)), p)
        print(self.print())
        return readable

    def calc_names(self):
        for node in self.nodes:
            action_names = []
            for action in node.path:
                action_names.append(action.name)
            node.name = "[" + ",".join(action_names) + "]"
            print(node.name)
        i = 0
        for infoset in self.infosets:
            if len(infoset.nodes) > 1:
                i = i + 1
                infoset.name = "I" + str(i)
            else:
                infoset.name = infoset.nodes[0].name

        self.variable_names = {}

        # check which action names are used multiple times
        lst = set()
        duplicate_action_names = set()
        for action in self.actions:
            if action.name in lst:
                duplicate_action_names.add(action.name)
            else:
                lst.add(action.name)

        for key in self.variable_map:
            variable = self.variable_map[key]
            if "b" in variable:
                self.variable_names[variable] = "B(" + key.name + "|" + key.infoset.name + ")"
            else:
                optional_infoset = ""
                if key.name in duplicate_action_names:
                    optional_infoset = "|" + key.infoset.name
                self.variable_names[variable] = "P(" + key.name + optional_infoset + ")"

    def calc_terminals(self):
        for node in self.nodes:
            if node.is_terminal:
                n = node
                n.terminals = [node]
                while not n.is_root:
                    n.parent.terminals.append(node)
                    n = n.parent
        self.terminals = self.root.terminals

    def calc_paths(self):
        frontier = [self.root]
        self.root.path = []
        while len(frontier) != 0:
            parent = frontier.pop(0)
            for action in parent.children:
                child = parent.children[action]
                child.path = parent.path + [action]
                frontier.append(child)

    def calc_variables(self):
        # also sorts self.infosets, self.nodes and self.actions
        self.infosets = []
        self.nodes = []
        self.actions = []
        self.variables = self.parameters.copy()
        self.variable_map = {}
        frontier = [self.root.infoset]
        c_i = 0
        while len(frontier) != 0:
            infoset = frontier.pop()
            self.players = max(self.players, infoset.player + 1)
            if infoset not in self.infosets:
                c_i += 1
                c_a = 0
                for action in infoset.actions:
                    c_a += 1
                    self.actions.append(action)
                    if infoset.is_chance():
                        self.variable_map[action] = action.prob
                    else:
                        var = "I" + str(c_i) + "A" + str(c_a) + "p"
                        self.variable_map[action] = var
                        self.variables.append(var)
                c_n = 0
                for node in infoset.nodes:
                    c_n += 1
                    self.nodes.append(node)
                    if not infoset.is_chance():
                        var = "I" + str(c_i) + "N" + str(c_n) + "b"
                        self.variable_map[node] = var
                        self.variables.append(var)
                    for child in node.children.values():
                        if child.is_terminal:
                            self.nodes.append(child)
                        else:
                            frontier.append(child.infoset)
                self.infosets.append(infoset)


class Infoset:

    def __init__(self):
        self.nodes = []
        self.actions = []
        self.player = -1
        self.terminals = []
        self.name = ""

    def is_chance(self):
        return self.player == -1

class Node:

    def __init__(self):
        self.parent = 0
        self.children = {}
        self.infoset = 0

        self.is_terminal = False
        self.outcome = []

        self.path = []
        self.is_root = False
        self.terminals = []

        self.name = ""

    def next(self, action):
        return self.children[action]


class Action:

    def __init__(self):
        self.infoset = 0
        self.prob = 0
        self.name = ""

    def is_chance(self):
        return self.infoset.is_chance()


class Solution:

    def __init__(self):
        self.type = ""
        self.utility = ""
        self.variable_constraints = {}