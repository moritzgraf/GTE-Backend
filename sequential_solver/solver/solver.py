import itertools
import pypolyhedron as poly
import numpy as np
import argparse
import time
import re
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

import subprocess

import sequential_solver.solver.gametree as gt

LOG = False


def read_ef(filename):
    f = open(filename, "r")
    g = gt.Game()
    eftg_nodes = {}
    eftg_actions = {}
    node_lines = []
    infoset_lines = []
    player_count = 0
    for line in f:
        entry = line.replace("\n", "").split(" ")
        print("line start: ", entry[0])
        if entry[0] == "player":
            player_count += 1
        elif entry[0] == "level":
            l = entry[1]
            loc = ""
            prev_loc = ""
            prev_action = ""
            payoff = ""
            player = ""
            i = 2
            while i < len(entry):
                if entry[i] == "node":
                    loc = l + "," + entry[i + 1]
                    i += 1
                elif entry[i] == "from":
                    prev_loc = entry[i + 1]
                    i += 1
                elif entry[i] == "move":
                    prev_action = entry[i + 1]
                    i += 1
                elif entry[i] == "payoffs":
                    payoff = []
                    for j in range(0, player_count):
                        payoff.append(entry[i + 1 + j])
                    i += 1
                elif entry[i] == "player":
                    player = entry[i + 1]
                    i += 1
                i += 1
            node_lines.append((loc, player, prev_loc, prev_action, payoff))
        elif entry[0] == "iset":
            locset = []
            i = 1
            while i < len(entry) and entry[i] != "player":
                locset.append(entry[i])
                i += 1
            player = entry[i + 1]
            infoset_lines.append((player, locset))
        elif entry[0] == "param":
            g.parameters.append(entry[1])
            eq = line.partition("restrict ")[2]
            if eq != "":
                g.equations.append(eq)
        elif entry[0] == "restrict":
            eq = line.partition("restrict ")[2]
            g.equations.append(eq)

    for nodeline in node_lines:
        loc, player, _, _, payoff = nodeline
        node = gt.Node()
        if len(payoff) != 0:
            node.outcome = payoff
            node.is_terminal = True

        g.nodes.append(node)
        eftg_nodes[loc] = node

    for isetline in infoset_lines:
        infoset = gt.Infoset()
        infoset.player = int(isetline[0]) - 1
        for loc in isetline[1]:
            infoset.nodes.append(eftg_nodes[loc])
            eftg_nodes[loc].infoset = infoset
        g.infosets.append(infoset)

    for nodeline in node_lines:
        loc, player, _, _, payoff = nodeline
        node = eftg_nodes[loc]
        if player != "" and node.infoset == 0:
            infoset = gt.Infoset()
            infoset.player = int(player) - 1
            infoset.nodes = [node]
            node.infoset = infoset
            g.infosets.append(infoset)


    for nodeline in node_lines:
        loc, player, prev_loc, prev_action, payoff = nodeline
        node = eftg_nodes[loc]
        if prev_loc == "":
            node.is_root = True
            g.root = node
        else:
            parent = eftg_nodes[prev_loc]
            action_id = str(parent.infoset) + ";" + prev_action
            if parent.infoset.player == -1:
                action_id += ";" + loc
            if action_id not in eftg_actions:
                action = gt.Action()
                g.actions.append(action)
                parent.infoset.actions.append(action)
                action.infoset = parent.infoset
                if parent.infoset.player == -1:
                    if "frac" in prev_action:
                        p, q = re.findall(r'(?<={).+?(?=})', prev_action)
                        action.prob = "(" + p + "/" + q + ")"
                    else:
                        action.prob = "(" + prev_action + ")"
                    action.name = "p" + str(len(parent.infoset.actions)) + action.prob
                else:
                    action.name = prev_action
                eftg_actions[action_id] = action

            node.parent = parent
            parent.children[eftg_actions[action_id]] = node

    g.process()
    f.close()
    return g


def read_efg(filename):
    g = gt.Game()
    f = open(filename, "r")
    efgtg_infosets = {}
    efgtg_actions = {}
    node_open_actions = {}
    node_stack = []
    for line in f:
        if len(line) == 0 or not re.match(r'[pct]', line):
            continue
        node_info, _, rest = line.partition("{")
        list_info, _, end = rest.partition("}")
        node_info_clean = re.sub(r'".+?"', "", node_info)
        numbers = re.findall(r'[0-9]', node_info_clean)
        list_strings = re.findall(r'".+?"', list_info)
        list_info_clean = re.sub(r'".+?"', "", list_info)
        list_numbers = re.findall(r'[0-9.]+', list_info_clean)
        node = gt.Node()
        g.nodes.append(node)
        if len(node_stack) == 0:
            node.is_root = True
            g.root = node
        else:
            parent = node_stack[-1]
            node.parent = parent
            prior_action = node_open_actions[parent].pop()
            parent.children[prior_action] = node
            if len(node_open_actions[parent]) == 0:
                node_stack.pop()

        if line[0] == "t":
            node.is_terminal = True
            for n in list_numbers:
                if "." in n:
                    decimal = len(n.partition(".")[2])
                    precision = "0." + ("0" * (decimal - 1)) + "1"
                    node.outcome.append("Rationalize[" + n + ", " + precision + "]")
                else:
                    node.outcome.append(n)
            node.outcome = list(list_numbers)
        else:
            player = -1
            infoset_id = ""
            if line[0] == "p":
                player = int(numbers[0]) - 1
                infoset_id = numbers[0] + "," + numbers[1]
            else:
                player = -1
                infoset_id = "c, " + numbers[0]

            if infoset_id not in efgtg_infosets:  #
                infoset = gt.Infoset()
                infoset.player = player
                g.infosets.append(infoset)
                efgtg_infosets[infoset_id] = infoset
                for i in range(len(list_strings)):
                    a = list_strings[i]
                    action = gt.Action()
                    g.actions.append(action)
                    infoset.actions.append(action)
                    action.infoset = infoset
                    if player == -1:
                        action.prob = list_numbers[i]
                    action_id = infoset_id + "," + a
                    efgtg_actions[action_id] = action
            infoset = efgtg_infosets[infoset_id]
            infoset.nodes.append(node)
            node.infoset = infoset
            node_open_actions[node] = list(infoset.actions)
            node_stack.append(node)
    g.process()
    f.close()
    return g


'''
def import_gambit(filename):
    gg = gambit.Game.read_game(filename)
    g = gt.game()
    g.players = range(0, len(gg.players))
    # translation dictionaries, gambit object to game object
    ggtg_infosets = {}
    ggtg_nodes = {}
    ggtg_actions = {}
    terminals = []
    g.root = gt.node()
    g.root.is_root = True
    g.nodes.append(g.root)
    ggtg_nodes[gg.root] = g.root

    frontier = [gg.root]
    i = 0
    while not len(frontier) == 0:
        i += 1
        if LOG : print("step ", i, " root has ", len(g.root.children), " children")
        parent = frontier.pop()
        node = ggtg_nodes[parent]
        if not parent.is_terminal:
            gambit_infoset = parent.infoset
            # if no matching infoset exists, create it first
            if gambit_infoset not in ggtg_infosets:
                infoset = gt.infoset()
                # add to translation dict
                ggtg_infosets[gambit_infoset] = infoset
                # add to game list
                g.infosets.append(infoset)
                # set player number, -1 for chance
                if gambit_infoset.is_chance:
                    infoset.player = -1
                else:
                    infoset.player = gambit_infoset.player.number
                # create actions from infoset
                for gambit_action in gambit_infoset.actions:
                    action = gt.action()
                    action.infoset = infoset
                    infoset.actions.append(action)
                    g.actions.append(action)
                    ggtg_actions[gambit_action] = action
                    if gambit_infoset.is_chance:
                        action.prob = gambit_action.prob
            # now link existing infoset and node
            ggtg_infosets[gambit_infoset].nodes.append(node)
            node.infoset = ggtg_infosets[gambit_infoset]
        else:
            for p in gg.players:
                node.outcome.append(str(parent.outcome[p]))
            node.is_terminal = True

        for child in parent.children:
            frontier.append(child)
            # create new node
            new_node = gt.node()
            # add to game list
            g.nodes.append(new_node)
            # set new nodes parent
            new_node.parent = node
            # set relevant entry of next in parent and prev in child
            node.children[ggtg_actions[child.prior_action]] = new_node
            # add new node to translation dict
            ggtg_nodes[child] = new_node

    g.calc_terminals()
    g.calc_paths()
    return g
'''


def extreme_directions_alt(A, timeout):
    start_time = time.time()
    if LOG: print("alternative method: ")
    if LOG: print(A)
    m, n = A.shape
    if LOG: print("cone dimensionality: ", n)
    doubleA = np.append(A, -A, axis=0)
    b = np.zeros(2 * m)
    p = poly.Hrep(doubleA, b)
    ed = []
    for g in p.generators:
        ed.append(g)
    if LOG: print("initial extreme directions found: ", len(ed))
    pairs = list(itertools.combinations(ed, 2))
    while len(pairs) != 0:
        if timeout != -1 and time.time() - start_time > timeout:
            return "Timeout"
        if LOG: print(len(pairs), len(ed))
        pair = pairs.pop(0)
        u, v = pair
        for i in range(len(u)):
            if u[i] * v[i] != 0:
                w = u[i] * v - v[i] * u
                new_pairs = []
                new = True
                if all(w == 0):
                    new = False
                else:
                    for e in ed:
                        if all(w == e) or all(w == -e):
                            new = False
                            break
                        else:
                            new_pairs.append((w, e))
                if new:
                    ed.append(w)
                    for p in new_pairs:
                        pairs.append(p)
    if LOG: print("list of extreme directions: ")
    for e in ed:
        if LOG: print(e)
    if LOG: print(time.time() - start_time)
    return ed


def extreme_directions_naive(A, timeout):
    start_time = time.time()
    if LOG: print("base method: ")
    if LOG: print(A)
    m, n = A.shape
    values = (0, 1, -1)
    if LOG: print("cone dimensionality: ", n)
    st = time.time()
    combinations = itertools.product(values, repeat=n)
    doubleA = np.append(A, -A, axis=0)
    b = np.zeros(2 * m + n)
    # ed_dict = {}
    # ed_dict_new = {}
    ed = []
    for comb in combinations:
        if timeout != -1 and time.time() - start_time > timeout:
            return "Timeout"
        # ed_dict[comb] = []
        # ed_dict_new[comb] = []
        C = np.diag(comb)
        finalA = np.append(doubleA, C, axis=0)
        p = poly.Hrep(finalA, b)
        for d in p.generators:
            # ed_dict[comb].append(d)
            new = True
            for d_old in ed:
                # equation is true iff vectors are multiples of each other
                if np.dot(d, d_old) * np.dot(d, d_old) == np.dot(d, d) * np.dot(d_old, d_old):
                    new = False
                    break
            if new:
                ed.append(d)
                # ed_dict_new[comb].append(d)
    et = time.time()
    if LOG: print("cone time: ", str(et - st))
    if LOG: print("list of extreme directions: ")
    for e in ed:
        if LOG: print(e)
    return ed  # , ed_dict, ed_dict_new


def extreme_directions_dd(A, timeout):
    start_time = time.time()
    if LOG: print("new method: ")
    if LOG: print(A)
    m, n = A.shape
    if LOG: print("cone dimensionality: ", n)
    doubleA = np.append(A, -A, axis=0)
    b = np.zeros(2 * m)
    p = poly.Hrep(doubleA, b)
    ed = []
    base_ed = []
    for g in p.generators:
        if LOG: print(g)
        ed.append(g)
        base_ed.append(g)
        base_ed.append(-g)
    if LOG: print("initial extreme directions found: ", len(ed))
    new_base_cones = {}
    new_base_cones[tuple([0] * n)] = list(base_ed)
    count = 1
    notnew = 0
    for i in range(n):

        if LOG: print(i, " base cones: ", len(new_base_cones), " instead of ", np.power(3, i))

        base_cones = new_base_cones
        new_base_cones = {}

        for base in base_cones:
            if timeout != -1 and time.time() - start_time > timeout:
                return "Timeout"
            count += 2
            U_old = base_cones[base]
            # if LOG : print("cone ", base, " has ", len(U_old), " extreme directions")
            U_pos = []
            U_neg = []
            U_zero = []
            for d in U_old:
                if d[i] > 0:
                    U_pos.append(d)
                elif d[i] < 0:
                    U_neg.append(d)
                else:
                    U_zero.append(d)
            U_new = []
            for u in U_pos:
                for v in U_neg:
                    if adjacent(base, U_old, u, v):
                        # if LOG : print(u, v, " adjacent in ", base)
                        w = u[i] * v - v[i] * u
                        if any(w != 0):
                            U_new.append(w)
                            new_ed = True
                            for e in ed:
                                if np.dot(e, w) * np.dot(e, w) == np.dot(e, e) * np.dot(w, w):
                                    new_ed = False
                                    if LOG: print("found extreme direction, not new:")
                                    if LOG: print(e, " matches ", w)
                                    notnew += 1
                                    break
                            if new_ed:
                                if LOG: print("found new extreme direction: ", w)
                                if LOG: print("from base ", base, " with new restriction at ", i)
                                if LOG: print("u = ", u)
                                if LOG: print("v = ", v)
                                ed.append(w)

            b_p = list(base)
            b_p[i] = 1
            U_new_pos = U_pos + U_zero + U_new
            if has_potential(b_p, U_new_pos, i + 1):
                new_base_cones[tuple(b_p)] = U_new_pos

            b_n = list(base)
            b_n[i] = -1
            U_new_neg = U_neg + U_zero + U_new
            if has_potential(b_n, U_new_neg, i + 1):
                new_base_cones[tuple(b_n)] = U_new_neg

            # if has_potential(base, U_old, i + 1):
            new_base_cones[base] = U_old

        if LOG: print(i, " total cones considered: ", count, " instead of ", np.power(3, i + 1))

    if LOG: print("list of extreme directions: ")
    for e in ed:
        if LOG: print(e)

    if LOG: print(time.time() - start_time)
    cone_max = np.power(3, n)
    if LOG: print(count, " of ", cone_max, " cones considered. (", int(count * 100 / cone_max), "%)")
    if LOG: print("notnew ", notnew)
    return ed


def active_constraints(b, x):
    return set(np.where(b != 0)[0]) & set(np.where(x == 0)[0])


def adjacent(b, ed, u, v):
    if all(np.array(b) == 0):
        return True
    z_u = active_constraints(b, u)
    z_v = active_constraints(b, v)
    if any(np.array(b) != 0):
        for w in ed:
            if all(w == u) or all(w == v):
                continue
            else:
                if (z_u & z_v) <= active_constraints(b, w):
                    return False
    return True


def has_potential(base, ed, i):
    if len(ed) < 2:
        return False
    M = np.row_stack(tuple(ed))
    for j in range(i, len(base)):
        if any(M[:, j] > 0) and any(M[:, j] < 0):
            return True

    return False


def equilibria_equations(g, include_sequential, restrict_belief, restrict_strategy, filter, ed_method, ed_timeout=-1):
    variable_map = g.variable_map.copy()  # <Node> : I1N3b, <Action> : I1A2p
    variables = g.variables.copy()
    sub_map = substitutions(g, variable_map)  # I1N3b : (1-I1N1b-I1N2b)
    resub_map = {}  # used later, I1N3b : I1N3b == (1-I1N1b-I1N2b)
    for key in variable_map:
        var = variable_map[key]
        if var in sub_map:
            variables.remove(var)
            variable_map[key] = sub_map[var]
            if not sub_map[var] == "(1)":
                resub_map[var] = var + " == " + sub_map[var]

    # belief variables are removed later
    variables_nash = variables.copy()

    equations_seq = []
    equations_nash = []

    # game utility, used in future calculations, and for evaluating the result
    if LOG: print("terminal probabilities: ")
    game_utility = [""] * g.players
    player_summands = []
    for i in range(g.players):
        player_summands.append([])
    for t in g.terminals:
        factors = []
        for a in t.path:
            factors.append(variable_map[a])
        product = "*".join(factors)
        if LOG: print(t, " ", product, " ", t.outcome)
        for i in range(g.players):
            player_summands[i].append(t.outcome[i] + "*" + product)

    equations_utility = []
    for i in range(g.players):
        game_utility[i] = "+".join(player_summands[i])
        equations_utility.append("P" + str(i + 1) + "u == " + game_utility[i])
        sub_map["P" + str(i + 1) + "u"] = game_utility[i]

    # basic rules
    # each action probability variable is positive
    for action in g.actions:
        if action.is_chance():
            continue
        var = variable_map[action]
        eq = ""
        if restrict_strategy:
            eq = var + "* (" + var + "-1) == 0"
        else:
            eq = var + " >= 0"
        equations_nash.append(eq)
        # equations_seq.append(eq)

    belief_vars = []
    # each node belief is positive, ignore chance sets, and only added in seqential equations
    for node in g.nodes:
        if node.is_terminal or node.infoset.is_chance():
            continue
        var = variable_map[node]
        eq = ""
        if restrict_belief:
            eq = var + "* (" + var + "-1) == 0"
        else:
            eq = var + " >= 0"
        equations_seq.append(eq)
        if var in variables_nash:
            variables_nash.remove(var)
            belief_vars.append(var)

    for g_eq in g.equations:
        nash_safe = True
        for var in belief_vars:
            if var in g_eq:
                nash_safe = False
                break
        pattern = re.compile(r'I\d+[AN]\d+[bp]|P\d+u')
        g_eq_sub = pattern.sub(
            lambda match: sub_map.get(match.group(0)) if match.group(0) in sub_map else match.group(0), g_eq)
        print("additional equation: ", nash_safe, g_eq_sub)
        if nash_safe:
            equations_nash.append(g_eq_sub)
            equations_seq.append(g_eq_sub)
        else:
            equations_seq.append(g_eq_sub)

    for infoset in g.infosets:
        if infoset.is_chance():
            continue
        # probabilities at each informationset sum to 1
        probs = []
        for action in infoset.actions:
            probs.append(variable_map[action])
        prob_eq = "+".join(probs) + "== 1"
        equations_nash.append(prob_eq)
        # equations_seq.append(prob_eq)
        # beliefs at each informationset sum to 1
        beliefs = []
        for node in infoset.nodes:
            beliefs.append(variable_map[node])
        belief_eq = "+".join(beliefs) + "== 1"
        equations_seq.append(belief_eq)

    if not include_sequential:
        equations_seq = ""
        variables = ""
    else:
        # sequential rationality
        for infoset in g.infosets:
            if infoset.is_chance():
                continue
            player = infoset.player
            # first express utility of playing action a in node n for all nodes and actions
            utility = {}
            for action in infoset.actions:
                utility[action] = {}
                for node in infoset.nodes:
                    summands = []
                    for tnode in node.next(action).terminals:
                        factors = [str(tnode.outcome[player])]
                        rem_actions = tnode.path[len(node.path) + 1:]
                        for a in rem_actions:
                            factors.append(variable_map[a])
                        product = "*".join(factors)
                        summands.append(product)
                    utility[action][node] = "(" + "+".join(summands) + ")"
            # express believed utility as weighted sum over all nodes and actions
            terms = []
            for action in infoset.actions:
                for node in infoset.nodes:
                    terms.append(variable_map[node] + "*" + variable_map[action] + "*" + utility[action][node])
            utility_at_I = "(" + "+".join(terms) + ")"

            # two equations per action at I:
            for action in infoset.actions:
                summands = []
                for node in infoset.nodes:
                    summands.append(variable_map[node] + "*" + utility[action][node])
                sum = "(" + "+".join(summands) + ")"
                # first equation: playing action in I is not better that Is utility
                equations_seq.append(sum + "-" + utility_at_I + "<= 0")
                # second equation: either prob of action or difference in utility is 0
                equations_seq.append(variable_map[action] + "(" + sum + "-" + utility_at_I + ") == 0")

        perfect_information = True
        for infoset in g.infosets:
            if len(infoset.nodes) > 1:
                perfect_information = False
                break

        if not perfect_information:
            # consistency
            # for consistency, construct a linear system that has approximate solutions iff the assessment is consistent
            alpha = []
            gamma = []
            for infoset in g.infosets:
                if LOG: print(infoset)
                if infoset.is_chance():
                    continue
                for node_pair in itertools.combinations(infoset.nodes, 2):
                    alpha.append(node_pair[0])
                    gamma.append(node_pair[1])
                    if LOG: print("node pair: ", node_pair)
            n_pairs = len(alpha)
            j_map = {}
            n = 0
            for action in g.actions:
                alpha.append(action)
                gamma.append("1")
                j_map[action] = n
                n += 1
            # n = len(g.actions)
            variable_map["1"] = "1"
            rows = []
            for p in range(n_pairs):
                h1, h2 = alpha[p], gamma[p]
                if LOG: print(h1, h2)
                row = [0] * n
                for a in h1.path:
                    row[j_map[a]] += 1
                for a in h2.path:
                    row[j_map[a]] += -1
                rows.append(row)
                if LOG: print(row)
            A = np.row_stack(tuple(rows))
            # if a column of A is zero everywhere, then that column can be removed
            # as can the corresponding action / 1 in alpha/gamma
            if LOG: print("shape of A before pruning: ", A.shape)
            if LOG: print(A)
            sa = ""
            for x in alpha:
                sa += variable_map[x] + ","
            if LOG: print(sa)
            sa = ""
            for x in gamma:
                sa += variable_map[x] + ","
            if LOG: print(sa)
            zero_columns = np.where(~A.any(axis=0))[0]
            A = np.delete(A, zero_columns, axis=1)
            alpha = list(np.delete(alpha, zero_columns + n_pairs, axis=0))
            gamma = list(np.delete(gamma, zero_columns + n_pairs, axis=0))
            if LOG: print("shape of A after pruning: ", A.shape)
            if LOG: print(A)
            sa = ""
            for x in alpha:
                sa += variable_map[x] + ","
            if LOG: print(sa)
            sa = ""
            for x in gamma:
                sa += variable_map[x] + ","
            if LOG: print(sa)
            A = np.append(A, np.identity(A.shape[1], ), axis=0)
            if LOG: print("final shape of A with identity: ", A.shape)
            # find extreme all relevant extreme directions of A
            eds = []
            if ed_method == "dd":
                eds = extreme_directions_dd(A.transpose(), ed_timeout)
            elif ed_method == "alt":
                eds = extreme_directions_alt(A.transpose(), ed_timeout)
            elif ed_method == "naive":
                eds = extreme_directions_naive(A.transpose(), ed_timeout)
            else:
                # default for wrong argument
                eds = extreme_directions_dd(A.transpose(), ed_timeout)
            if eds == "Timeout":
                return "Timeout"
            # each extreme direction defines an equation
            if LOG: print("alpha: ", alpha)
            if LOG: print("gamma: ", gamma)
            for ed in eds:
                if LOG: print("ed")
                left = []
                right = []
                for i in range(len(ed)):
                    p = int(ed[i])
                    if p > 0:
                        left.append(variable_map[alpha[i]] + "^" + str(p))
                        right.append(variable_map[gamma[i]] + "^" + str(p))
                    if p < 0:
                        right.append(variable_map[alpha[i]] + "^" + str(-p))
                        left.append(variable_map[gamma[i]] + "^" + str(-p))
                if LOG: print(left, " == ", right)
                equations_seq.append(" * ".join(left) + " == " + " * ".join(right))

    # nash rationality
    nash_strategies = []
    if filter == "weak":
        # following version of nash rationality only considers single deviations, not any deviation
        # resulting eqilibria are not necessarily nash equilibria, resulting in a weaker filter
        # but number of equations is no longer exponential in number of actions
        for infoset in g.infosets:
            if infoset.is_chance():
                continue
            player = infoset.player
            # represents the utility of all terminal nodes after I
            terminal_utilities = []
            # represents the utility of all terminal nodes after I if action a is played with prob 1
            mod_terminal_utilities = {}
            for action in infoset.actions:
                mod_terminal_utilities[action] = []
            for node in infoset.nodes:
                for tnode in node.terminals:
                    factors = [str(tnode.outcome[player])]
                    for a in tnode.path:
                        factors.append(variable_map[a])
                    product = "*".join(factors)
                    terminal_utilities.append(product)
                    for action in infoset.actions:
                        if action in tnode.path:
                            mod_product = product.replace(variable_map[action], "1")
                            mod_terminal_utilities[action].append(mod_product)

            base_utiltiy = "(" + "+".join(terminal_utilities) + ")"
            for action in infoset.actions:
                mod_utility = "(" + "+".join(mod_terminal_utilities[action]) + ")"
                # two equations for every action:
                utility_diff = "(" + mod_utility + "-" + base_utiltiy + ")"
                # if the action is played, utility difference is 0
                equations_nash.append(variable_map[action] + "*" + utility_diff + "==0")
                # utility difference is always <= 0
                equations_nash.append(utility_diff + "<= 0")
    elif filter == "full":

        player_infoset_actions = []
        for i in range(g.players):
            player_infoset_actions.append([])

        if LOG: print(g.players)
        for infoset in g.infosets:
            if infoset.player != -1:
                player_infoset_actions[infoset.player].append(infoset.actions)

        nash_strategies = [[]] * (g.players + 1)
        for i in range(g.players):
            nash_strategies[i] = itertools.product(*player_infoset_actions[i])

        for i in range(g.players):
            player_utility = game_utility[i]
            for strategy in nash_strategies[i]:
                mod_utility = player_utility
                product = []
                for action in strategy:
                    mod_utility = mod_utility.replace(variable_map[action], "1")
                    for a in action.infoset.actions:
                        if not a == action:
                            mod_utility = mod_utility.replace(variable_map[a], "0")
                    product.append(variable_map[action])
                if len(product) == 0:
                    continue
                s_prob = "*".join(product)
                if LOG: print("player ", i, " strategy: ", s_prob)
                # 2 equations per pure strategy:
                # playing pure strategy does not have more utility than equilibrium strategy
                equations_nash.append("(" + mod_utility + ")-(" + player_utility + ") <= 0")
                # if strategy is played, it has the same utility as equilibrium strategy
                equations_nash.append(s_prob + "*((" + mod_utility + ")-(" + player_utility + ")) == 0")
    elif filter == "alt":
        # nash strategies are pure strategies of that player that have a different set of possible outcomes
        # can be obtained from combinations of actions of that player in terminal histories

        nash_strategies = []
        for i in range(g.players):
            nash_strategies.append([])
        for tnode in g.terminals:
            for i in range(g.players):
                action_set = set()
                for a in tnode.path:
                    if a.infoset.player == i:
                        action_set.add(a)
                if len(action_set) == 0:
                    continue
                new = True
                for j in range(len(nash_strategies[i])):
                    if action_set.issubset(nash_strategies[i][j]):
                        new = False
                        break
                    elif nash_strategies[i][j].issubset(action_set):
                        new = False
                        nash_strategies[i][j] = action_set
                        break
                if new:
                    nash_strategies[i].append(action_set)
        for i in range(g.players):
            player_utility = game_utility[i]
            for strategy in nash_strategies[i]:
                mod_utility = player_utility
                product = []
                for action in strategy:
                    mod_utility = mod_utility.replace(variable_map[action], "1")
                    for a in action.infoset.actions:
                        if not a == action:
                            mod_utility = mod_utility.replace(variable_map[a], "0")
                    product.append(variable_map[action])
                if len(product) == 0:
                    continue
                s_prob = "*".join(product)
                if LOG: print("player ", i, " strategy: ", s_prob)
                # 2 equations per pure strategy:
                # playing pure strategy does not have more utility than equilibrium strategy
                equations_nash.append("(" + mod_utility + ")-(" + player_utility + ") <= 0")
                # if strategy is played, it has the same utility as equilibrium strategy
                equations_nash.append(s_prob + "*((" + mod_utility + ")-(" + player_utility + ")) == 0")

    if LOG: print(len(equations_nash), " equations for nash equilibria")
    if LOG: print(len(equations_seq), " equations for sequential equilibria")
    equations_seq = " && ".join(equations_seq)
    equations_nash = " && ".join(equations_nash)
    equations_utility = " && ".join(equations_utility)

    return equations_seq, equations_nash, equations_utility, variable_map, resub_map, variables, variables_nash


def substitutions(g, variable_map):
    substitute_map = {}
    # simplify out variables that are always 1  or last one of every information set
    for infoset in g.infosets:
        if infoset.is_chance():
            continue
        lst = ["1"]
        for node in infoset.nodes:
            lst.append(variable_map[node])
        sub = "(" + "-".join(lst[:-1]) + ")"
        # substitute_map[infoset.nodes[-1]] = sub
        substitute_map[variable_map[infoset.nodes[-1]]] = sub
        lst = ["1"]
        for action in infoset.actions:
            lst.append(variable_map[action])
        sub = "(" + "-".join(lst[:-1]) + ")"
        # substitute_map[infoset.actions[-1]] = sub
        substitute_map[variable_map[infoset.actions[-1]]] = sub

    return substitute_map


def wolfram_solve_equations(g, result, include_se, include_ne, session=None):
    eq, eq_n, eq_u, var_map, resub_map, var, var_n = result

    if not session:
        session = WolframLanguageSession()
    redef_inputform = '''\nUnprotect[Inequality];\n
                Format[HoldPattern @ Inequality[a__], InputForm] := Module[{res = HoldForm[a], rel},
                    rel = List @@ Replace[res[[2 ;; -1 ;; 2]],
                        {
                        Less -> " < ",
                        LessEqual -> " <= ",
                        Greater -> " > ",
                        GreaterEqual -> " >= ",
                        Equal -> " == ",
                        Unequal -> " != "
                        },
                        {1}
                    ];
                    res = InputForm /@ res;
                    res[[2 ;; -1 ;; 2]] = rel;
                    Replace[res, HoldForm[z__] :> OutputForm @ HoldForm @ SequenceForm[z]]
                    ]\n
                Protect[Inequality];\n'''
    session.evaluate(wlexpr(redef_inputform))
    expr_var = "{" + ", ".join(var) + "}"
    ne_expr_var = "{" + ", ".join(var_n) + "}"
    nash_call = "CylindricalDecomposition[Rationalize[" + eq_n + ", 0], " + ne_expr_var + "]"
    nash_solutions = ()
    if include_ne:
        session.evaluate(wlexpr("{nashtime, nashresult} = AbsoluteTiming[" + nash_call + "]"))
        session.evaluate(wlexpr("nes = BooleanConvert[Simplify[nashresult]]"))
        session.evaluate(wlexpr("NE = If[Head[nes] == Or, List @@ nes, {nes}, {nes}]"))
        nash_solutions = session.evaluate(wlexpr("Map[Function[x, ToString[x, InputForm]], NE]"))
        if LOG: print("Nash Solutions: ", nash_solutions, len(nash_solutions))
    seq_solutions = ()
    if include_se:
        seq_call = "CylindricalDecomposition[Rationalize[" + eq_n + " && " + eq + ", 0] , " + expr_var + "]"
        session.evaluate(wlexpr("{seqtime, seqresult} = AbsoluteTiming[" + seq_call + "]"))
        session.evaluate(wlexpr("newEQ = BooleanConvert[Simplify[seqresult]]"))
        session.evaluate(wlexpr("SE = If[Head[newEQ] == Or, List @@ newEQ, {newEQ}, {newEQ}]"))
        seq_solutions = session.evaluate(wlexpr("Map[Function[x, ToString[x, InputForm]], SE]"))

    wolfram_solutions = nash_solutions + seq_solutions
    solutions = []
    for i in range(len(wolfram_solutions)):
        nash = i < len(nash_solutions)
        solution = gt.Solution()
        vars = []
        expr_vars = ""
        if nash:
            solution.type = "Nash Equilibria"
            vars = var_n
            expr_vars = ne_expr_var
        else:
            solution.type = "Sequential Equilibria"
            vars = var
            expr_vars = expr_var
        # obtain clean solution by additional decompositon (gets rid of 3x == 1 type equations that Simplify created)
        solution_eq = session.evaluate(
            wlexpr("ToString[CylindricalDecomposition[Rationalize[" + wolfram_solutions[i] + ", 0], " + expr_vars + "], InputForm]"))

        constraints = solution_eq.split(" && ")
        # because of the variable ordering in the decomposition
        # the variable assigned in c is the variable v in c that is furthest in the variable order
        for c in constraints:
            for v in vars[::-1]:
                if v in c:
                    solution.variable_constraints[v] = c
                    break

        # resubstitutions are done with an additional cylindrical decomposition
        for sub_var in resub_map:
            if "b" in sub_var and nash:
                continue
            if LOG: print("resub of " + sub_var)
            expr_vars = "{" + ", ".join(vars + [sub_var]) + "}"
            call = "subresult = CylindricalDecomposition[Rationalize[" + solution_eq + " && " + resub_map[
                sub_var] + ", 0], " + expr_vars + "]"
            session.evaluate(wlexpr(call))
            # session.evaluate(wlexpr("resub = BooleanConvert[Simplify[subresult]]"))
            resub_solution = session.evaluate(wlexpr("ToString[subresult, InputForm]"))
            if LOG: print(resub_solution)
            for c in resub_solution.split(" && "):
                if sub_var in c:
                    solution.variable_constraints[sub_var] = c
                    break

        # calculations of utilities also with additional cylindrical decomposition#
        var_u = []
        for j in range(g.players):
            var_u.append("P" + str(j + 1) + "u")
        expr_var_u = "{" + ", ".join(vars + var_u) + "}"
        utility_call = "CylindricalDecomposition[Rationalize[" + solution_eq + " && " + eq_u + ", 0], " + expr_var_u + "]"
        session.evaluate(wlexpr("{utilitytime, utilityresult} = AbsoluteTiming[" + utility_call + "];"))
        # session.evaluate(wlexpr("utility = BooleanConvert[Simplify[utilityresult]];"))
        utility_solution = session.evaluate(wlexpr("ToString[utilityresult, InputForm]"))

        solution.utility = []
        for i in range(g.players):
            v = "P" + str(i + 1) + "u"
            for c in utility_solution.split(" && "):
                if v in c:
                    a, b, = c.split(" == ", 1)
                    if a == v:
                        solution.utility.append(b)
                    elif b == v:
                        solution.utility.append(a)
                    else:
                        solution.utility.append(c)
        solutions.append(solution)

    session.terminate()
    return solutions


def write_wolframscript(result, output_file_name, header, pytime, mformat):
    eq, eq_n, utility, var_map, sub_map, var, var_n = result

    f = open(output_file_name, "w")
    f.truncate()
    f.write("#!/usr/bin/env wolframscript")
    f.write('''\nUnprotect[Inequality];\n
                Format[HoldPattern @ Inequality[a__], InputForm] := Module[{res = HoldForm[a], rel},
                    rel = List @@ Replace[res[[2 ;; -1 ;; 2]],
                        {
                        Less -> " < ",
                        LessEqual -> " <= ",
                        Greater -> " > ",
                        GreaterEqual -> " >= ",
                        Equal -> " == ",
                        Unequal -> " != "
                        },
                        {1}
                    ];
                    res = InputForm /@ res;
                    res[[2 ;; -1 ;; 2]] = rel;
                    Replace[res, HoldForm[z__] :> OutputForm @ HoldForm @ SequenceForm[z]]
                    ]\n
                Protect[Inequality];\n''')
    f.write("\nPrint[\"" + header + "\"]")

    f.write("\nSE = {}")
    f.write("\nNE = {}")
    f.write("\nsetime = 0")
    expr_var = "{" + ", ".join(var) + "}"
    ne_expr_var = "{" + ", ".join(var_n) + "}"
    nash_call = "CylindricalDecomposition[" + eq_n + ", " + ne_expr_var + ", \"Function\"]"
    f.write("\n{nashtime, nashresult} = AbsoluteTiming[" + nash_call + "]")
    f.write("\nnes = BooleanConvert[Normal[nashresult]];")
    f.write("\nNE = If[Head[nes] == Or, List @@ nes, {nes}, {nes}];")
    extra_rest = " nashresult && "

    call = "CylindricalDecomposition[" + extra_rest + eq + ", " + expr_var + "]"
    f.write("\n{time, result} = AbsoluteTiming[" + call + "];")
    f.write("\n newEQ = BooleanConvert[Simplify[result]];")
    f.write("\nSE = Join[SE, If[Head[newEQ] == Or, List @@ newEQ, {newEQ}, {newEQ}]];")

    f.write("\nPrint[\"Equations generated in \", " + pytime + ", \"s (python)\"]")
    f.write("\nPrint[\"\n\nNash Equilibria, calculated in \", nashtime, \"s (mathematica):\"]")
    f.write("\nFor[i=1,i<=Length[NE],i++, Print[Format[NE[[i]], " + mformat + "]]]")
    f.write("\n \nPrint[\"\n\nSequential Equilibria, calculated in \", time, \"s (mathematica): \"]\n")
    f.write(
        "\nFor[i=1,i<=Length[SE],i++, Print[Format[CylindricalDecomposition[SE[[i]], " + expr_var + "], " + mformat + "]]]")
    f.write("\nPrint[\"\nTotal required time: \", (" + pytime + "+ nashtime + time), \"s\"]")
    f.close()


def convert_to_gte(g, s):
    pattern2 = re.compile(r'Nash Equilibria[^:]*:\s*([\S\s]*)Sequential Equilibria[^:]*:\s*([\S\s]*)Total required')
    match = pattern2.search(s)
    nash_text = match.group(1).replace(" && ", ", ")
    seq_text = match.group(2).replace(" && ", ", ")
    gametree_text = g.print()
    metadata_text = ""
    seperator = "\n-----\n"
    gte_text = "Nash Equilibria:\n" + nash_text + "Sequential Equilibria:\n" + seq_text
    gte_text += seperator + "Game Tree:\n" + gametree_text
    gte_text += seperator + "" + metadata_text
    return gte_text


def test_import(g):
    if LOG: print("#nodes: ", len(g.nodes))
    if LOG: print("#infosets ", len(g.infosets))
    if LOG: print("#actions ", len(g.actions))
    for node in g.nodes:
        if LOG: print(node)
        if LOG: print("has ", len(node.children), " children")
        if LOG: print("and ", len(node.terminals), " terminals")
        if LOG: print("and outcome ", node.outcome)
    for infoset in g.infosets:
        if LOG: print(infoset, "belonging to player ", infoset.player)
        if LOG: print("is chance?", infoset.is_chance())
        if LOG: print("has nodes ", infoset.nodes)
        if LOG: print("and actions ", infoset.actions)


def solve(g,

          output_file="",
          create_wls=False,
          long_output=False,
          include_header=False,
          include_time=False,

          include_nash=False,
          include_sequential=False,
          restrict_belief=False,
          restrict_strategy=False,
          filter="full",
          extreme_directions="dd"):
    s = ""
    if include_header:
        header = "Analysis of Equilibria in " + file_name + "\n"
        if not include_sequential:
            header += "Analysis of Nash Equilibria"
        elif not include_nash:
            header += "Analysis of Sequential Equilibria"
        else:
            header += "Analysis of Nash Equilibria and Sequential Equilibria"
        if restrict_belief:
            header += "\n beliefs restricted to {0, 1}"
        if restrict_strategy:
            header += "\n strategies restricted to {0, 1}"
        if filter == "weak":
            header += "\n filtered using only local deviation nash equilibria"
        elif filter == "full":
            header += "\n filtered using full nash rationality"
        elif filter == "alt":
            header += "\n filtered using full nash rationality (alternative method)"
        s += header + "\n\n"

    start_time = time.time()
    equations = equilibria_equations(g,
                                     restrict_belief=restrict_belief,
                                     restrict_strategy=restrict_strategy,
                                     include_sequential=include_sequential,
                                     filter=filter,
                                     ed_method=extreme_directions)
    eq_time = time.time()
    if create_wls:
        if not output_file:
            output_file = "solution.wls"
        write_wolframscript(equations, output_file, header, str(eq_time - start_time), "InputForm")
    else:
        g.solutions = wolfram_solve_equations(g, equations, include_sequential, include_nash)
        solve_time = time.time()
        include_types = []
        if include_nash:
            include_types.append("Nash Equilibria")
        if include_sequential:
            include_types.append("Sequential Equilibria")
        s += g.print_solutions(long=long_output, include_types=include_types)
        if include_time:
            s += "\n\n Calculated in " + str(solve_time - start_time) + "s:\n"
            s += str(eq_time - start_time) + " to build equations\n"
            s += str(solve_time - eq_time) + " to solve equations\n"
        if output_file:
            f = open(output_file, "w")
            f.write(s)
        return s


def solve_from_file(file_name,
                    output_file="",
                    create_wls=False,
                    long_output=False,
                    include_header=False,
                    include_time=False,

                    include_nash=False,
                    include_sequential=False,
                    restrict_belief=False,
                    restrict_strategy=False,
                    filter="full",
                    extreme_directions="dd"):
    g = import_game(file_name)

    return solve(g, output_file=output_file, create_wls=create_wls, long_output=long_output,
                 include_header=include_header, include_time=include_time, include_nash=include_nash,
                 include_sequential=include_sequential, restrict_belief=restrict_belief,
                 restrict_strategy=restrict_strategy, filter=filter, extreme_directions=extreme_directions)


def import_game(file_name):
    fn_split = file_name.split(".")
    g = gt.Game()
    try:
        if len(fn_split) != 0:
            if fn_split[1] == "efg":
                # g = import_gambit(file_name)
                g = read_efg(file_name)
            elif fn_split[1] == "ef":
                g = read_ef(file_name)
    except KeyError:
        parser = argparse.ArgumentParser()
        parser.error("filename " + args.file_name + " is not a valid file type (use .ef or .efg)")
    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name",
                        help="name of a .efg file containing the game you want to analyze")
    parser.add_argument("-o", "--output",
                        help="specify the file name for the output"
                             "required to end in .txt\n",
                        type=str,
                        default="")
    parser.add_argument("-eq", "--equilibria",
                        help="specify which equilibria to calculate and list in the result:\n"
                             "'ne' to calculate only Nash Equilibria\n"
                             "'seq to only show Sequential Equilibria (NE are still calculated)\n"
                             "'full' to calculate and show both (default)",
                        type=str,
                        default="full")
    # changes to the calculations
    parser.add_argument("-rb", "--restrict_belief",
                        help="restricts to equilibria where all beliefs are in {0, 1}",
                        action="store_true")
    parser.add_argument("-rs", "--restrict_strategy",
                        help="restricts to equilibria where all action probabilities are in {0, 1}",
                        action="store_true")

    parser.add_argument("-f", "--filter",
                        help="choose method to compute all extreme directions:\n"
                             "'full' to filter using regular nash equilibria\n"
                             "'weak' to filter using weak nash equilibria (only single action deviations)\n"
                             "'alt' to filter using an experimental method (should be equivalent to ne)",
                        type=str,
                        default="ne")
    parser.add_argument("-ed", "--extreme_directions",
                        help="choose method to compute all extreme directions:\n"
                             "'dd' to use the modified double description with pruning (default)\n"
                             "'alt' to use the alternative method of matching pairs\n"
                             "'naive' to iterate over all cones",
                        type=str,
                        default="dd")
    # changes to the formating
    parser.add_argument("-l", "--long",
                        help="makes the output more descriptive",
                        action="store_true")
    parser.add_argument("-t", "--time",
                        help="include the calculation time in the output",
                        action="store_true")
    parser.add_argument("-c", "--config",
                        help="include a header that describes the sovlers configuration",
                        action="store_true")

    args = parser.parse_args()
    file_name = args.file_name
    output_file = args.output
    long_output = args.long
    include_header = args.config
    include_time = args.time
    include_nash = args.equilibria in ('ne', 'full')
    include_sequential = args.equilibria in ('seq', 'full')
    restrict_belief = args.restrict_belief
    restrict_strategy = args.restrict_strategy
    filter = args.filter
    extreme_directions = args.extreme_directions
    # create wls: write wls file for later use instead of solving it
    # currently not supported
    create_wls = False

    s = solve_from_file(file_name, output_file=output_file, create_wls=create_wls, long_output=long_output,
                        include_header=include_header, include_time=include_time, include_nash=include_nash,
                        include_sequential=include_sequential, restrict_belief=args.restrict_belief,
                        restrict_strategy=args.restrict_strategy, filter=args.filter,
                        extreme_directions=args.extreme_directions)

    '''

    if file_name_txt:
        text = subprocess.check_output(["wolframscript", file_name_wls])
        s = text.decode()
        pattern = re.compile(r'I\d+[AN]\d+[bp]')
        readable = pattern.sub(lambda match: g.variable_names.get(match.group(0)), s)
        if LOG : print(readable)
        f2 = open(file_name_txt, "w")
        f2.truncate()
        f2.write(readable)
        f2.close()
        if file_name_gte:
            text = convert_to_gte(g, readable)
            if LOG : print(text)
            f3 = open(file_name_gte, "w")
            f3.truncate()
            f3.write(text)
            f3.close()
    '''
