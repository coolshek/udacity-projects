
from sample_players import DataPlayer
import random, math
# from isolation import DebugState


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        if len (state.actions()) == 1:
            # dbstate = DebugState.from_state(state)
            # print (dbstate)
            self.queue.put(state.actions()[0])
            return
        if state.ply_count < 2:
            action = random.choice(state.actions())
        else:
            action = uct_search(state).action
            # action = iterative_deepening(state, self.player_id, 5)
        # dbstate = DebugState.from_state(state)
        # print (dbstate)
        if action is None:
            print("Incorrect action")
            action = random.choice(state.actions())
        self.queue.put(action)

# A policy is a mapping from states to actions, specifying which action will be
# chosen from each state in S. The aim is to find the policy that yields the highest expected reward.

# Game theory extends decision theory to situations in which multiple agents interact. A game can be defined
# as a set of established rules that allows the interaction of one or more players to produce specified outcomes.

# MONTE CARLO TREE SEARCH
# This section introduces the family of algorithms known
# as Monte Carlo Tree Search (MCTS). MCTS rests on two
# fundamental concepts: that the true value of an action
# may be approximated using random simulation; and that
# these values may be used efficiently to adjust the policy
# towards a best-first strategy. The algorithm progressively
# builds a partial game tree, guided by the results of previous
# exploration of that tree. The tree is used to estimate
# the values of moves, with these estimates (particularly
# those for the most promising moves) becoming more
# accurate as the tree is built.

# Algorithm
# The basic algorithm involves iteratively building a search
# tree until some predefined computational budget – typically
# a time, memory or iteration constraint – is reached,
# at which point the search is halted and the best-performing
# root action returned. Each node in the search
# tree represents a state of the domain, and directed links
# to child nodes represent actions leading to subsequent
# states.

# Four steps are applied per search iteration:
#   1) Selection: Starting at the root node, a child selection
#   policy is recursively applied to descend through
#   the tree until the most urgent expandable node is
#   reached. A node is expandable if it represents a non-terminal
#   state and has unvisited (i.e. unexpanded) children.
#   2) Expansion: One (or more) child nodes are added to
#   expand the tree, according to the available actions.
#   3) Simulation: A simulation is run from the new node(s)
#   according to the default policy to produce an outcome.
#   4) Back propagation: The simulation result is "backed
#   up" (i.e. back propagated) through the selected
#   nodes to update their statistics.

# These may be grouped into two distinct policies:
#   1) Tree Policy: Select or create a leaf node from the
#   nodes already contained within the search tree (selection
#   and expansion).
#   2) Default Policy: Play out the domain from a given
#   non-terminal state to produce a value estimate (simulation).

class Node():
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.action = action
        self.reward = 0
        self.visit_count = 1
        self.parent = parent
        self.actions_done = []
        self.children = []

budget = 25

def uct_search(state):
    root_node = Node(state)
    for i in range(budget):
        node = tree_policy(root_node)
        if node is None:
            continue
        reward = default_policy(node.state)
        backup_nega_max(node, reward)
    return best_child(root_node)

def tree_policy(node):
    while not node.state.terminal_test():
        if not fully_expanded(node):
            return expand(node)
        else:
            exploration_constant = 1 / math.sqrt(2)
            node = best_child(node, exploration_constant)
    return node

def expand(node):
    action_list = list(set(node.state.actions()) - set(node.actions_done))
    if len(action_list) == 0:
        return None
    action = action_list[0]
    changed_state = node.state.result(action)
    child = Node(changed_state, node, action)
    node.children.append(child)
    node.actions_done.append(action)
    return node.children[-1]

def fully_expanded(node):
    return len(node.state.actions()) == len(node.actions_done)

def best_child(node, exploration_constant=0.5):
    best_children = []
    best_score = float('-inf')
    for child in node.children:
        exploitation = child.reward / child.visit_count
        exploration = exploration_constant *  math.sqrt( 2 * math.log(node.visit_count) / child.visit_count )
        score = exploitation + exploration
        if best_score < score:
            best_children = [child]
            best_score = score
        elif best_score == score:
            best_children.append(child)
    return random.choice(best_children)

def default_policy(state):
    player = state.player()
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))
    # return -1 if state._has_liberties(player) else 1
    return -1 if any(state.liberties(state.locs[player])) else 1

def backup_nega_max(node, reward):
    while node is not None:
        node.visit_count += 1
        node.reward += reward
        reward = -reward
        node = node.parent


# Baseline Iterative deepening Alpha-beta search

def iterative_deepening(state, player, depth = 3):
    best_move = None
    for d in range(1, depth + 1):
        best_move = alpha_beta_search(state, player, d)
    if best_move is None:
        best_move = random.choice(state.actions())
    return best_move


def alpha_beta_search(game_state, player, depth):
    """ Return the move along a branch of the game tree that
    has the best possible value.  A move is a pair of coordinates
    in (column, row) order corresponding to a legal move for
    the searching player.

    You can ignore the special case of calling this function
    from a terminal state.
    """
    alpha = float("-inf")
    beta = float("inf")
    best_score = float("-inf")
    best_move = None

    # TODO: modify the function signature to accept an alpha and beta parameter
    def min_value(state, min_alpha, min_beta, min_depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(0)
        if min_depth <=0 :
            return score(state)

        v = float("inf")
        for a in state.actions():
            v = min(v, max_value(state.result(a), min_alpha, min_beta, min_depth - 1))
            if v <= min_alpha:
                return v
            min_beta = min(min_beta, v)
        return v


    # TODO: modify the function signature to accept an alpha and beta parameter
    def max_value(state, max_alpha, max_beta, max_depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(0)
        if max_depth <=0 :
            return score(state)

        v = float("-inf")
        for a in state.actions():
            v = max(v, min_value(state.result(a), max_alpha, max_beta, max_depth - 1))
            if v >= max_beta:
                return v
            max_alpha = max(max_alpha, v)
        return v


    def score(state):
        own_loc = state.locs[player]
        opp_loc = state.locs[1 - player]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    for a in game_state.actions():
        vv = min_value(game_state.result(a), alpha, beta, depth)
        alpha = max(alpha, vv)
        if vv > best_score:
            best_score = vv
            best_move = a
    return best_move