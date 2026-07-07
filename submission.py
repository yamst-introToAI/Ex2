import time
import random
import math
# import ipdb
from func_timeout import func_timeout, FunctionTimedOut

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance


EXPECTIMAX_ACTION_WEIGHTS = {
    "move north": 4,
    "charge": 4,
}

TIE_BREAKING_ORDER = [
    "drop off",
    "pick up",
    "charge",
    "move north",
    "move east",
    "move south",
    "move west",
    "park",
]


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)

    #  Feature 1: Credit difference
    credit_diff = robot.credit - other_robot.credit

    #  Feature 2: Distance to target (package or destination)
    dist_to_target = 0
    if robot.package is not None:
        #  If holding a package, the target is the packages destination
        dist_to_target = manhattan_distance(robot.position, robot.package.destination)
    else:
        #  If empty-handed, find the closest package currently on the board
        packages_on_board = [p for p in env.packages if p.on_board]
        if packages_on_board:
            distances = [manhattan_distance(robot.position, p.position) for p in packages_on_board]
            dist_to_target = min(distances)
        else:
            dist_to_target = 0

    #  Feature 3a: Battery level
    battery = min(robot.battery, env.num_steps / 2)
    
    # Feature 3b: Define a low battery tax that especially penalizes very low battery levels
    low_battery_tax = max(0, 10 - 0.5 * battery * battery)
    
    # Feature 4: Bonus for having a package in hand, to help in a "flat" search space.
    package_bonus = 1 + manhattan_distance(robot.package.position, robot.package.destination) if robot.package is not None else 0
    
    #  Calculate final heuristic value
    return (2 * credit_diff) + (4 * battery) + (package_bonus) - (dist_to_target) - (low_battery_tax) 

################ DECISION FUNCTIONS ################

def pick_any_action(env: WarehouseEnv, robot_id: int):
    """
    When any legal action is needed to be picked fast, this returns the first in TIE_BREAKING_ORDER.
    """
    moves = env.get_legal_operators(robot_id)

    moves.sort(key=lambda op: TIE_BREAKING_ORDER.index(op) if op in TIE_BREAKING_ORDER else 999)
    return moves[0]

def _successors_mm(env: WarehouseEnv, robot_id: int):
    operators = env.get_legal_operators(robot_id)
    operators.sort(key=lambda op: TIE_BREAKING_ORDER.index(op) if op in TIE_BREAKING_ORDER else 999)
    children = [env.clone() for _ in operators]
    for child, op in zip(children, operators):
        child.apply_operator(robot_id, op)
    return operators, children

def minimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    heuristic_fn = heuristic_fn or smart_heuristic
    if depth <= 0:
        return pick_any_action(env, robot_id)

    max_value = -math.inf
    best_op = None
    for operator, state in zip(*_successors_mm(env, robot_id)):
        next_robot_id = (robot_id + 1) % 2
        value = _minimax_min_node(state, next_robot_id, depth - 1, heuristic_fn, robot_id)
        if value > max_value:
            max_value = value
            best_op = operator
    return best_op

def _minimax_max_node(env: WarehouseEnv, current_robot_id: int, depth: int, heuristic_fn, original_robot_id: int,):
    if depth == 0 or env.done():
        return heuristic_fn(env, original_robot_id)

    max_value = -math.inf
    for _, state in zip(*_successors_mm(env, current_robot_id)):
        next_robot_id = (current_robot_id + 1) % 2
        value = _minimax_min_node(state, next_robot_id, depth - 1, heuristic_fn, original_robot_id)
        max_value = max(max_value, value)
    return max_value


def _minimax_min_node(env: WarehouseEnv, current_robot_id: int, depth: int, heuristic_fn, original_robot_id: int):
    if depth == 0 or env.done():
        return heuristic_fn(env, original_robot_id)

    min_value = math.inf
    for _, state in zip(*_successors_mm(env, current_robot_id)):
        next_robot_id = (current_robot_id + 1) % 2
        value = _minimax_max_node(state, next_robot_id, depth - 1, heuristic_fn, original_robot_id)
        min_value = min(min_value, value)
    return min_value

def alphabeta_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    heuristic_fn = heuristic_fn or smart_heuristic
    if depth <= 0:
        return pick_any_action(env, robot_id)

    max_value = -math.inf
    best_op = None
    alpha = -math.inf
    beta = math.inf

    for operator, state in zip(*_successors_mm(env, robot_id)):
        next_robot_id = (robot_id + 1) % 2
        value = _alphabeta_min_node(state, next_robot_id, depth - 1, heuristic_fn, robot_id, alpha, beta)
        if value > max_value:
            max_value = value
            best_op = operator
        alpha = max(alpha, max_value)
    return best_op


def _alphabeta_max_node(env: WarehouseEnv, current_robot_id: int, depth: int, heuristic_fn, original_robot_id: int,
                        alpha: float, beta: float):
    if depth == 0 or env.done():
        return heuristic_fn(env, original_robot_id)

    max_value = -math.inf
    for _, state in zip(*_successors_mm(env, current_robot_id)):
        next_robot_id = (current_robot_id + 1) % 2
        value = _alphabeta_min_node(state, next_robot_id, depth - 1, heuristic_fn, original_robot_id, alpha, beta)
        max_value = max(max_value, value)
        alpha = max(alpha, max_value)
        if max_value >= beta:
            return max_value  # Prune
    return max_value

def _alphabeta_min_node(env: WarehouseEnv, current_robot_id: int, depth: int, heuristic_fn, original_robot_id: int,
                        alpha: float, beta: float):
    if depth == 0 or env.done():
        return heuristic_fn(env, original_robot_id)

    min_value = math.inf
    for _, state in zip(*_successors_mm(env, current_robot_id)):
        next_robot_id = (current_robot_id + 1) % 2
        value = _alphabeta_max_node(state, next_robot_id, depth - 1, heuristic_fn, original_robot_id, alpha, beta)
        min_value = min(min_value, value)
        beta = min(beta, min_value)
        if min_value <= alpha:
            return min_value  # Prune
    return min_value

def _successors(env: WarehouseEnv, robot_id: int):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        return operators, children

def expectimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited expectimax.
    The opponent's legal actions are weighted by EXPECTIMAX_ACTION_WEIGHTS;
    every legal action not in the dictionary has weight 1.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    heuristic_fn = heuristic_fn or smart_heuristic
    if depth <= 0:
        return pick_any_action(env, robot_id)
    # max of successors, selected by CHOOSE_ORDER
    actions_values = [] # Action, value tuples
    # iterate over every state and operator in the successors of the current state using zip
    for operator, state in zip(*_successors(env, robot_id)):
        next_robot_id = (robot_id + 1) % 2
        value = _expectimax_value_expectation_node(state, next_robot_id, depth - 1, heuristic_fn)
        actions_values.append((operator, value))

    # Pick first action in TIE_BREAKING_ORDER that has the max value
    max_value = max(value for _, value in actions_values)
    best_actions = [action for action, value in actions_values if value == max_value]
    for action in TIE_BREAKING_ORDER:
        if action in best_actions:
            return action

def _expectimax_value_max_node(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn):
    """
    Return the value of a max node in the expectimax tree.
    Calculated as the maximum value of the possible actions of robot_id."""
    if depth == 0 or env.done():
        return heuristic_fn(env, robot_id)
    max_value = None
    for _, state in zip(*_successors(env, robot_id)):
        next_robot_id = (robot_id + 1) % 2
        value = _expectimax_value_expectation_node(state, next_robot_id, depth - 1, heuristic_fn)
        max_value = value if max_value is None else max(max_value, value)

    assert max_value is not None, "There is always a legal operator."
    return max_value

def _expectimax_value_expectation_node(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn):
    """
    Return the value of an expectation node in the expectimax tree.
    Calculated as the expected value based on the possible actions of robot_id and the action weights.
    """
    if depth == 0 or env.done():
        return heuristic_fn(env, (robot_id + 1) % 2) # This is rival. Heuristic calculated for other robot (us).
    total_value = 0

    operators = env.get_legal_operators(robot_id)
    total_weight = sum(EXPECTIMAX_ACTION_WEIGHTS.get(operator, 1) for operator in operators)
    assert total_weight > 0, "There is always a legal operator."

    for op, state in zip(*_successors(env, robot_id)):
        next_robot_id = (robot_id + 1) % 2
        value = _expectimax_value_max_node(state, next_robot_id, depth - 1, heuristic_fn)
        weight = EXPECTIMAX_ACTION_WEIGHTS.get(op, 1)
        total_value += weight * value
    return total_value / total_weight

################ DECISION FUNCTIONS ################

########## AGENTS #########
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        limit = min(time.time() + time_limit - 0.16, time.time() + 2)
        # Running for very long is redundant. Heuristic is based on randomized package spawns and not more accurate.
        if time_limit < 0.01:
            return pick_any_action(env, agent_id)
        else:
            best_action = minimax_decision(env, agent_id, 1, smart_heuristic)

        try:
            depth = 2
            while time.time() < limit:
                args = (env, agent_id, depth, smart_heuristic)
                op = func_timeout(limit - time.time(), minimax_decision, args=args)
                best_action = op or best_action
                depth += 1
        except FunctionTimedOut:
            pass

        return best_action


class AgentAlphaBeta(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        limit = min(time.time() + time_limit - 0.16, time.time() + 2)
        # Running for very long is redundant. Heuristic is based on randomized package spawns and not more accurate.
        if time_limit < 0.01:
            return pick_any_action(env, agent_id)
        else:
            best_action = alphabeta_decision(env, agent_id, 1, smart_heuristic)

        try:
            depth = 2
            while time.time() < limit:
                args = (env, agent_id, depth, smart_heuristic)
                op = func_timeout(limit - time.time(), alphabeta_decision, args=args)
                best_action = op or best_action
                depth += 1
        except FunctionTimedOut:
            pass

        return best_action

class AgentExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        limit = min(time.time() + time_limit - 0.16, time.time() + 4)
        # Running for very long is redundant. Heuristic is based on randomized package spawns and not more accurate.
        if time_limit < 0.01:
            return pick_any_action(env, agent_id)
        else:
            best_action = expectimax_decision(env, agent_id, 1, smart_heuristic)

        depth = 2
        while limit > time.time():
            try:
                args = (env, agent_id, depth, smart_heuristic)
                best_action = func_timeout(limit - time.time(), expectimax_decision, args=args)
                depth += 1
            except FunctionTimedOut:
                break
        return best_action

# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
