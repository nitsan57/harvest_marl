import cv2
from collections import defaultdict
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utility_funcs as utils
import gym
from gym.utils import seeding

import random

import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import os


HARVEST_DEFAULT_VIEW_SIZE = 5
TIMEOUT_TIME = 25

HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

ACTIONS_DICT = {'MOVE_LEFT': (-1, 0),  # Move left
                'MOVE_RIGHT': (1, 0),  # Move right
                'MOVE_UP': (0, -1),  # Move up
                'MOVE_DOWN': (0, 1),  # Move down
                'STAY': (0, 0),  # don't move
                # Rotate counter clockwise
                'TURN_CLOCKWISE': ((0, -1), (1, 0)),
                'TURN_COUNTERCLOCKWISE': ((0, 1), (-1, 0)),
                'FIRE': 99}  # fire code_vec_dummy

FIRE_LEN = 5
HUMAN_ACTION_TO_INDEX = {'MOVE_LEFT': 0,  # Move left
                         'MOVE_RIGHT': 1,  # Move right
                         'MOVE_UP': 2,  # Move up
                         'MOVE_DOWN': 3,  # Move down
                         'STAY': 4,  # don't move
                         # Rotate counter clockwise
                         'TURN_CLOCKWISE': 5,
                         'TURN_COUNTERCLOCKWISE': 6,
                         'FIRE': 7}  # length of firing range


INDEX_TO_HUMAN_ACTION = {v: k for k, v in HUMAN_ACTION_TO_INDEX.items()}

INDEX_TO_ACTION_VEC = {k: ACTIONS_DICT[v]
                       for k, v in INDEX_TO_HUMAN_ACTION.items()}


ACTION_TO_HUMAN = {v: k for k, v in ACTIONS_DICT.items()}


ORIENTATIONS = {'LEFT': [-1, 0],
                'RIGHT': [1, 0],
                'UP': [0, -1],
                'DOWN': [0, 1]}

DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background
                   '': [180, 180, 180],  # Grey board walls
                   '@': [180, 180, 180],  # Grey board walls
                   'A': [0, 255, 0],  # Green apples
                   'F': [255, 255, 0],  # Yellow fining beam
                   'P': [159, 67, 255],  # Purple player

                   # Colours for agents. R value is a unique identifier
                   '0': [255, 255, 255],  # Black background beyond map walls
                   '1': [159, 67, 255],  # Purple
                   '2': [2, 81, 154],  # Blue
                   '3': [204, 0, 204],  # Magenta
                   '4': [216, 30, 54],  # Red
                   '5': [254, 151, 0],  # Orange
                   '6': [100, 255, 255],  # Cyan
                   '7': [99, 99, 255],  # Lavender
                   '8': [250, 204, 255],  # Pink
                   '9': [238, 223, 16]}  # Yellow


SPAWN_PROB = [0, 0.005, 0.02, 0.05]
OUTCAST_POSITION = -99
HARVEST_DEFAULT_VIEW_SIZE = 5
APPLE_RADIUS = 2


class HarvestMap (MultiAgentEnv):
    metadata = {'render.modes': [
        'human', 'rgb_array'], 'video.frames_per_second': 2}

    def __init__(self, num_agents, ascii_map=HARVEST_MAP, agent_view_range=HARVEST_DEFAULT_VIEW_SIZE, fire_len=FIRE_LEN, render=True, color_map=None):
        self.run_steps = 200
        self.curr_step = 0
        self.viewer = None

        self.ID_LIST = [str(i) for i in range(num_agents)]

        self.num_agents = num_agents
        self.fire_len = fire_len
        self.agent_view_range = agent_view_range
        self.apple_points = []

        # NB: Ray throws exceptions for any `0` value Discrete
        # observations so we'll make position a 1's based value
        # self.observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=(2 * self.agent_view_range + 1,
        #                                                                   2 * self.agent_view_range + 1, 3), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(42,
                                                                        42, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(8)

        self.base_map = utils.ascii_to_numpy(ascii_map)
        # map without agents or beams
        self.world_map = np.full(
            (len(self.base_map), len(self.base_map[0])), ' ')

        self.beam_pos = []

        # returns the agent at a desired position if there is one
        self.color_map = color_map if color_map is not None else DEFAULT_COLOURS
        self.spawn_points = []  # where agents can appear
        self.wall_points = []

        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append((row, col))
                elif self.base_map[row, col] == '@':
                    self.wall_points.append([row, col])
                elif self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])

        self.agents = {}
        self.setup_agents()

        # self.reset()

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        self.curr_step = 0
        self.beam_pos = []
        self.agents = {}
        self.setup_agents()
        self.reset_map()
        self.map_events()

        observations = {}
        map_with_agents = self.get_map_with_agents()
        for agent_id, agent in self.agents.items():
            position = agent['position']
            rgb_arr = utils.map_to_colors(utils.return_view(map_with_agents, position,
                                                            self.agent_view_range, self.agent_view_range), self.color_map)
            rgb_arr = cv2.resize(rgb_arr.astype(
                np.uint8), (42, 42), interpolation=cv2.INTER_AREA)  # only for test
            observations[agent_id] = rgb_arr
        return observations

    def reset_map(self):
        self.world_map = np.full(
            (len(self.base_map), len(self.base_map[0])), ' ')
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.world_map[row, col] = '@'
        for apple_point in self.apple_points:
            self.world_map[apple_point[0], apple_point[1]] = 'A'

    def setup_agents(self):
        """Construct all the agents for the environment"""

        for i in range(self.num_agents):
            agent_id = self.ID_LIST[i]
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # When hit, agent is cast away from map for `remaining_timeout` n_steps

            self.agents[agent_id] = {
                'position': spawn_point, 'orientation': rotation, 'remaining_timeout': 0}

    def spawn_rotation(self):
        """Return a randomly selected initial rotation for an agent"""
        rand_int = np.random.randint(len(ORIENTATIONS.keys()))
        return list(ORIENTATIONS.keys())[int(rand_int)]

    def spawn_point(self):
        """Returns a randomly selected spawn point."""
        spawn_index = 0
        is_free_cell = False
        chosen_positions = []
        random.shuffle(self.spawn_points)
        for i, spawn_point in enumerate(self.spawn_points):
            if (spawn_point[0], spawn_point[1]) not in chosen_positions:
                spawn_index = i
                is_free_cell = True
                chosen_positions.append((spawn_point[0], spawn_point[1]))
        assert is_free_cell, 'There are not enough spawn points! Check your map?'
        return self.spawn_points[spawn_index]

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        map_with_agnets = self.get_map_with_agents()
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if map_with_agnets[row, col] not in self.ID_LIST and map_with_agnets[row, col] != 'A':
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if 0 <= x + j < self.world_map.shape[0] and self.world_map.shape[1] > y + k >= 0:
                                symbol = self.world_map[x + j, y + k]
                                if symbol == 'A':
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = np.random.rand(1)[0]
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, 'A'))
        return new_apple_points

    def update_map(self, new_points):
        """For points in new_points, place desired char on the map"""
        for i in range(len(new_points)):
            row, col, char = new_points[i]
            self.world_map[row, col] = char

    def map_events(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

        # Outcast timed-out agents
        for agent_id, agent in self.agents.items():
            if agent['remaining_timeout'] > 0:
                agent['remaining_timeout'] -= 1
                # print("Agent %s its on timeout for %d n_steps" % (agent_id, agent.remaining_timeout))
                if not np.any(agent['position'][0] == OUTCAST_POSITION):
                    self.update_map(
                        [[agent['position'][0], agent['position'][1], ' ']])
                    agent['position'] = (OUTCAST_POSITION, OUTCAST_POSITION)
            # Return agent to environment
            if agent['remaining_timeout'] == 0 and agent['position'][0] == OUTCAST_POSITION:
                # print("%s has finished timeout" % agent_id)
                spawn_point = self.spawn_point()
                spawn_rotation = self.spawn_rotation()
                self.agents[agent_id] = {
                    'position': spawn_point, 'orientation': spawn_rotation, 'remaining_timeout': 0}

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get('A', 0)
        return num_apples

    def get_map_with_agents(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.

        Returns:
            2D array of strings representing the map.
        """
        grid = np.copy(self.world_map)

        for agent_id, agent in self.agents.items():
            char_id = agent_id

            # If agent is not within map, skip.
            if not(agent['position'][0] >= 0 and agent['position'][0] < grid.shape[0] and
                    agent['position'][1] >= 0 and agent['position'][1] < grid.shape[1]):
                continue

            grid[agent['position'][0], agent['position'][1]] = char_id

        for beam_pos in self.beam_pos:
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]

        return grid

    def step(self, actions):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        observations = {}
        rewards = defaultdict(int)
        dones = {}
        info = {}

        self.beam_pos = []

        # filter outcast players
        actions_temp = {}
        for agent_id, agent in self.agents.items():
            pos = agent['position']
            if pos[0] == -99:
                continue
            actions_temp[agent_id] = actions[agent_id]

        actions = actions_temp

        self.update_moves(actions)

        for agent_id, agent in self.agents.items():
            pos = agent['position']
            if pos[0] == -99:
                rewards[agent_id] = rewards[agent_id]
                continue
            new_char, reward = self.consume(self.world_map[pos[0], pos[1]])
            rewards[agent_id] = rewards[agent_id] + reward
            self.world_map[pos[0], pos[1]] = new_char

        # execute custom moves like firing

        self.update_special_actions(actions, rewards)  # inside update moves
        self.map_events()  # step of desidgend env

        # execute spawning events

        map_with_agents = self.get_map_with_agents()

        for agent_id, agent in self.agents.items():
            # pos = agent['position']
            # if pos[0] == -99:
            #     rewards[agent_id] = rewards[agent_id]
            #     continue
            agent_view = utils.return_view(
                map_with_agents, agent['position'], self.agent_view_range, self.agent_view_range)
            rgb_arr = utils.map_to_colors(agent_view, self.color_map)
            rgb_arr = self.rotate_view(agent['orientation'], rgb_arr)
            rgb_arr = cv2.resize(rgb_arr.astype(
                np.uint8), (42, 42), interpolation=cv2.INTER_AREA)  # only for test
            observations[agent_id] = rgb_arr
            dones[agent_id] = False  # no final state here!
        dones["__all__"] = np.any(list(dones.values()))
        if self.curr_step >= self.run_steps:
            dones["__all__"] = True
        self.curr_step += 1

        return [observations, rewards, dones, info]

    def update_moves(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions.
        Also resolves conflicts over multiple agents wanting a cell.

        This method works by finding all conflicts over a cell and randomly assigning them
       to one of the agents that desires the slot. It then sets all of the other agents
       that wanted the cell to have a move of staying. For moves that do not directly
       conflict with another agent for a cell, but may not be temporarily resolvable
       due to an agent currently being in the desired cell, we continually loop through
       the actions until all moves have been satisfied or deemed impossible.
       For example, agent 1 may want to move from [1,2] to [2,2] but agent 2 is in [2,2].
       Agent 2, however, is moving into [3,2]. Agent-1's action is first in the order so at the
       first pass it is skipped but agent-2 moves to [3,2]. In the second pass, agent-1 will
       then be able to move into [2,2].

        Parameters
        ----------
        agent_actions: dict
            dict with agent_id as key and action as value
        """
        map_with_agents = self.get_map_with_agents()

        reserved_slots = []
        for agent_id, action in agent_actions.items():
            action_human = INDEX_TO_HUMAN_ACTION[action]
            agent = self.agents[agent_id]
            selected_action = INDEX_TO_ACTION_VEC[action]
            # TODO(ev) these two parts of the actions
            if 'MOVE' in action_human or 'STAY' in action_human:
                # rotate the selected action appropriately
                rot_action = self.rotate_action(
                    selected_action, agent['orientation'])
                new_pos = tuple(
                    np.array(agent['position']) + np.array(rot_action))
                # allow the agents to confirm what position they can move to
                new_pos = self.return_valid_pos(agent, new_pos)
                reserved_slots.append((*new_pos, 'P', agent_id))
            elif 'TURN' in action_human:
                new_rot = self.update_rotation(
                    action_human, agent['orientation'])
                agent['orientation'] = new_rot

        # now do the conflict resolution part of the process

        # helpful for finding the agent in the conflicting slot
        agent_by_pos = {tuple(agent['position']):
                        agent_id for agent_id, agent in self.agents.items()}

        # agent moves keyed by ids
        agent_moves = {}

        # lists of moves and their corresponding agents
        move_slots = []
        agent_to_slot = []

        for slot in reserved_slots:
            row, col = slot[0], slot[1]
            if slot[2] == 'P':
                agent_id = slot[3]
                agent_moves[agent_id] = (row, col)
                move_slots.append((row, col))
                agent_to_slot.append(agent_id)

        # cut short the computation if there are no moves
        if len(agent_to_slot) > 0:
            # first we will resolve all slots over which multiple agents
            # want the slot
            # shuffle so that a random agent has slot priority
            shuffle_list = list(zip(agent_to_slot, move_slots))
            np.random.shuffle(shuffle_list)  # inplace shuffle
            agent_to_slot, move_slots = zip(*shuffle_list)
            unique_move, indices, return_count = np.unique(move_slots, return_index=True,
                                                           return_counts=True, axis=0)
            search_list = move_slots

            # first go through and remove moves that can't possible happen. Three types
            # 1. Trying to move into an agent that has been issued a stay command
            # 2. Trying to move into the spot of an agent that doesn't have a move
            # 3. Two agents trying to walk through one another

            # Resolve all conflicts over a space
            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        # check that the cell you are fighting over doesn't currently
                        # contain an agent that isn't going to move for one of the agents
                        # If it does, all the agents commands should become STAY
                        # since no moving will be possible
                        move = tuple(move)
                        conflict_indices = np.where(move in search_list)[0]
                        all_agents_id = [agent_to_slot[i] for i in conflict_indices]
                        # all other agents now stay in place so update their moves
                        # to reflect this
                        conflict_cell_free = True
                        for agent_id in all_agents_id:
                            moves_copy = agent_moves.copy()
                            # TODO(ev) code duplication, simplify
                            row, col = move
                            if map_with_agents[row, col] in self.ID_LIST:
                                # find the agent that is currently at that spot and make sure
                                # that the move is possible. If it won't be, remove it.
                                conflicting_agent_id = agent_by_pos[move]
                                curr_pos = self.agents[agent_id]['position']
                                curr_conflict_pos = self.agents[conflicting_agent_id]['position']
                                conflict_move = agent_moves.get(conflicting_agent_id,
                                                                curr_conflict_pos)
                                # Condition (1):
                                # a STAY command has been issued
                                if agent_id == conflicting_agent_id:
                                    conflict_cell_free = False
                                # Condition (2)
                                # its command is to stay
                                # or you are trying to move into an agent that hasn't
                                # received a command

                                elif conflicting_agent_id not in moves_copy.keys() or curr_conflict_pos == conflict_move:
                                    conflict_cell_free = False

                                # Condition (3)
                                # It is trying to move into you and you are moving into it
                                elif conflicting_agent_id in moves_copy.keys():
                                    if agent_moves[conflicting_agent_id] == curr_pos and \
                                            move == self.agents[conflicting_agent_id]['position']:
                                        conflict_cell_free = False

                        # if the conflict cell is open, let one of the conflicting agents
                        # move into it
                        if conflict_cell_free:
                            self.agents[agent_to_slot[index]
                                        ]['position'] = move
                            agent_by_pos = {tuple(agent['position']):
                                            agent_id for agent_id, agent in self.agents.items()}
                        # ------------------------------------
                        # remove all the other moves that would have conflicted
                        remove_indices = np.where(move in search_list)[0]

                        all_agents_id = [agent_to_slot[i]
                                         for i in remove_indices]
                        # all other agents now stay in place so update their moves
                        # to stay in place
                        for agent_id in all_agents_id:
                            agent_moves[agent_id] = self.agents[agent_id]['position']

            # make the remaining un-conflicted moves
            agent_by_pos = {agent['position']: agent_id for agent_id, agent in self.agents.items()}
            # print("AGNET BY POS", agent_by_pos)
            while len(agent_moves.items()) > 0:

                num_moves = len(agent_moves.items())
                moves_copy = agent_moves.copy()
                del_keys = []
                for agent_id, move in moves_copy.items():
                    if agent_id in del_keys:
                        continue
                    if move[0] != -99 and map_with_agents[move[0], move[1]] in self.ID_LIST:
                        # find the agent that is currently at that spot and make sure
                        # that the move is possible. If it won't be, remove it.
                        # print("YYYYYYYYYY", move, agent_by_pos)
                        conflicting_agent_id = agent_by_pos[move]
                        curr_pos = self.agents[agent_id]['position']
                        curr_conflict_pos = self.agents[conflicting_agent_id]['position']
                        conflict_move = agent_moves.get(
                            conflicting_agent_id, curr_conflict_pos)
                        # Condition (1):
                        # a STAY command has been issued
                        if agent_id == conflicting_agent_id:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (2)
                        # its command is to stay
                        # or you are trying to move into an agent that hasn't received a command
                        elif conflicting_agent_id not in moves_copy.keys() or \
                                curr_conflict_pos == conflict_move:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (3)
                        # It is trying to move into you and you are moving into it
                        elif conflicting_agent_id in moves_copy.keys():

                            if agent_moves[conflicting_agent_id] == curr_pos and move == self.agents[conflicting_agent_id]['position']:
                                del agent_moves[conflicting_agent_id]
                                del agent_moves[agent_id]
                                del_keys.append(agent_id)
                                del_keys.append(conflicting_agent_id)
                    # this move is unconflicted so go ahead and move
                    else:
                        self.agents[agent_id]['position'] = move
                        del agent_moves[agent_id]
                        del_keys.append(agent_id)

                # no agent is able to move freely, so just move them all
                # no updates to hidden cells are needed since all the
                # same cells will be covered
                if len(agent_moves) == num_moves:
                    for agent_id, move in agent_moves.items():
                        self.agents[agent_id]['position'] = move
                    break

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            return ' ', 1
        else:
            return char, 0

    def update_special_actions(self, agent_actions, rewards):
        for agent_id, action in agent_actions.items():
            human_action = INDEX_TO_HUMAN_ACTION[action]
            # check its not a move based action
            if 'MOVE' not in human_action and 'STAY' not in human_action and 'TURN' not in human_action:
                agent = self.agents[agent_id]
                if agent['position'][0] == -99:
                    continue
                updates = self.special_action(agent, human_action, rewards)
                if len(updates) > 0:
                    self.update_map(updates)

    def special_action(self, agent, action, rewards):
        # agent.fire_beam('F')
        #     fire_beam(self, char):
        # if char == 'F':
        #     self.reward_this_turn -= 0

        updates = self.update_map_fire(agent['position'],
                                       agent['orientation'],
                                       self.fire_len, rewards, fire_char='F')
        return updates

    def update_map_fire(self, firing_pos, firing_orientation, fire_len, rewards, fire_char, cell_types=[],
                        update_char=[], blocking_cells='P'):
        """From a firing position, fire a beam that may clean or hit agents
        Notes:
            (1) Beams are blocked by agents
            (2) A beam travels along until it hits a blocking cell at which beam the beam
                covers that cell and stops
            (3) If a beam hits a cell whose character is in cell_types, it replaces it with
                the corresponding index in update_char
            (4) As per the rules, the beams fire from in front of the agent and on its
                sides so the beam that starts in front of the agent travels out one
                cell further than it does along the sides.
            (5) This method updates the beam_pos, an internal representation of how
                which cells need to be rendered with fire_char in the agent view
        Parameters
        ----------
        firing_pos: (list)
            the row, col from which the beam is fired
        firing_orientation: (list)
            the direction the beam is to be fired in
        fire_len: (int)
            the number of cells forward to fire
        fire_char: (str)
            the cell that should be placed where the beam goes
        cell_types: (list of str)
            the cells that are affected by the beam
        update_char: (list of str)
            the character that should replace the affected cells.
        blocking_cells: (list of str)
            cells that block the firing beam
        Returns
        -------
        updates: (tuple (row, col, char))
            the cells that have been hit by the beam and what char will be placed there
        """
        agent_by_pos = {
            tuple(agent['position']): agent_id for agent_id, agent in self.agents.items()}
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        # compute the other two starting positions
        right_shift = self.rotate_right(firing_direction)
        firing_pos = [start_pos, start_pos + right_shift - firing_direction,
                      start_pos - right_shift - firing_direction]
        firing_points = []
        updates = []
        map_with_agents = self.get_map_with_agents()
        for pos in firing_pos:
            next_cell = pos + firing_direction
            # check if the cell blocks beams. For example, waste blocks beams.
            if self.world_map[next_cell[0], next_cell[1]] in blocking_cells:
                break
            for i in range(fire_len):
                if self.test_if_in_bounds(next_cell) and \
                        self.world_map[next_cell[0], next_cell[1]] != '@':

                    # FIXME(ev) code duplication
                    # agents absorb beams
                    # activate the agents hit function if needed
                    if map_with_agents[next_cell[0], next_cell[1]] in self.ID_LIST:
                        agent_id = agent_by_pos[(next_cell[0], next_cell[1])]
                        self.hit_fire(agent_id, rewards, fire_char)
                        firing_points.append(
                            (next_cell[0], next_cell[1], fire_char))
                        if self.world_map[next_cell[0], next_cell[1]] in cell_types:
                            type_index = cell_types.index(self.world_map[next_cell[0],
                                                                         next_cell[1]])
                            updates.append(
                                (next_cell[0], next_cell[1], update_char[type_index]))
                        break

                    # update the cell if needed
                    if self.world_map[next_cell[0], next_cell[1]] in cell_types:
                        type_index = cell_types.index(
                            self.world_map[next_cell[0], next_cell[1]])
                        updates.append(
                            (next_cell[0], next_cell[1], update_char[type_index]))

                    firing_points.append(
                        (next_cell[0], next_cell[1], fire_char))

                    # check if the cell blocks beams. For example, waste blocks beams.
                    # if self.world_map[next_cell[0], next_cell[1]] in blocking_cells:
                    #     break

                    # increment the beam position
                    next_cell += firing_direction

                else:
                    break

        self.beam_pos += firing_points
        return updates

    def hit_fire(self, agent_id, rewards, char):
        if char == 'F':  # double check...remove if possible after stable
            # no reward change if there is - put here
            rewards[agent_id] = rewards[agent_id] - 0
            if self.agents[agent_id]['remaining_timeout'] == 0:
                # print("%s was hit with timeout beam" % self.agent_id)
                self.agents[agent_id]['remaining_timeout'] = TIMEOUT_TIME

    def return_valid_pos(self, agent, new_pos):
        if agent['remaining_timeout'] > 0:
            return agent['position']
        else:
            ego_new_pos = new_pos
            new_row, new_col = ego_new_pos
            # you can't walk through walls
            temp_pos = new_pos
            map_with_agents = self.get_map_with_agents()
            if map_with_agents[new_row, new_col] == '@':
                temp_pos = agent['position']
            return temp_pos

    def rotate_action(self, action_vec, orientation):
        # WARNING: Note, we adopt the physics convention that \theta=0 is in the +y direction
        if orientation == 'UP':
            return action_vec
        elif orientation == 'LEFT':
            return self.rotate_left(action_vec)
        elif orientation == 'RIGHT':
            return self.rotate_right(action_vec)
        else:
            return self.rotate_left(self.rotate_left(action_vec))

    def rotate_left(self, action_vec):
        return tuple(np.dot(ACTIONS_DICT['TURN_COUNTERCLOCKWISE'], action_vec))

    def rotate_right(self, action_vec):
        return tuple(np.dot(ACTIONS_DICT['TURN_CLOCKWISE'], action_vec))

    # TODO(ev) this should be an agent property
    def update_rotation(self, human_action, curr_orientation):
        if human_action == 'TURN_COUNTERCLOCKWISE':
            if curr_orientation == 'LEFT':
                return 'DOWN'
            elif curr_orientation == 'DOWN':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'UP'
            else:
                return 'LEFT'
        else:
            if curr_orientation == 'LEFT':
                return 'UP'
            elif curr_orientation == 'UP':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'DOWN'
            else:
                return 'LEFT'

    def rotate_view(self, orientation, view):
        """Takes a view of the map and rotates it the agent orientation
        Parameters
        ----------
        orientation: str
            str in {'UP', 'LEFT', 'DOWN', 'RIGHT'}
        view: np.ndarray (row, column, channel)
        Returns
        -------
        a rotated view
        """
        if orientation == 'UP':
            return view
        elif orientation == 'LEFT':
            return np.rot90(view, k=1, axes=(0, 1))
        elif orientation == 'DOWN':
            return np.rot90(view, k=2, axes=(0, 1))
        elif orientation == 'RIGHT':
            return np.rot90(view, k=3, axes=(0, 1))
        else:
            raise ValueError('Orientation {} is not valid'.format(orientation))

    # TODO(ev) this definitely should go into utils or the general agent class
    def test_if_in_bounds(self, pos):
        """Checks if a selected cell is outside the range of the map"""
        if pos[0] < 0 or pos[0] >= self.world_map.shape[0]:
            return False
        elif pos[1] < 0 or pos[1] >= self.world_map.shape[1]:
            return False
        else:
            return True

    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """

        map_with_agents = self.get_map_with_agents()
        rgb_arr = utils.map_to_colors(map_with_agents, self.color_map)
        im = rgb_arr

        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=500)
            scale = im.shape[0] / im.shape[1]
            human_im = cv2.resize(
                im.astype(np.uint8), (500, int(500*scale)), interpolation=cv2.INTER_AREA)
            self.viewer.imshow(human_im)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return np.asarray(im)

    def close_viewer(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if self.viewer:
            self.viewer.close()
