import time
# import ipdb
from func_timeout import func_timeout, FunctionTimedOut

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


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

    #  Feature 3: Battery level
    battery = robot.battery
    package_bonus = 10 if robot.package is not None else 0

    #  Calculate final heuristic value
    return (3 * credit_diff) + (0 * battery) + (package_bonus) - (dist_to_target)


# TODO: section b : fixed-depth helper for deterministic grading
def minimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited minimax.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    raise NotImplementedError()


# TODO: section c : fixed-depth helper for deterministic grading
def alphabeta_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited alpha-beta pruning.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    raise NotImplementedError()

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
    assert depth > 0 and not env.done()
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

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 4
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        time_limit -= 0.1
        time_remaining = time_limit
        best_action = expectimax_decision(env, agent_id, 1, smart_heuristic)
        # start running expectimax with increasing depth until time runs out.
        depth = 2
        while time_remaining > 0:
            try:
                time_remaining = time_limit - (time.time() - start)
                best_action = func_timeout(time_remaining, expectimax_decision, args=(env, agent_id, depth, smart_heuristic))
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
