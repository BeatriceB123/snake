import math

import numpy as np
import os
import datetime
import pygame
import random
from keras.optimizers import SGD, RMSprop
from ple import PLE
from ple.games.snake import Snake
from keras import Sequential, layers, optimizers
from collections import deque

WIDTH = 400
HEIGHT = 400

NO_SENSORS = 0

original_env = os.environ.copy()

my_directions = {'sus': 0, 'stanga': 1, 'dreapta': 2, 'jos': 3}
my_directions_coord = {'sus': [0, -1], 'stanga': [-1, 0], 'dreapta': [1, 0], 'jos': [0, 1]}

oposite = {'sus': 'jos', 'stanga': 'dreapta', 'dreapta': 'stanga', 'jos': 'sus'}


def interface(with_interface=False):
    if not with_interface:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


class DQNAgent:

    def __init__(self, p, game, rewards):
        self.p = p
        self.game = game
        self.rewards = rewards

        self.directie = 'dreapta'
        self.previous_state = {}
        self.actions = p.getActionSet()

        self.model = None

        self.train_frames = 0
        self.observe = 1_000
        self.no_frames_to_save = 0
        self.no_frames_between_trains = 0

        self.epsilon = 0.99  # exploration rate
        self.gamma = 0.9  # discount rate
        self.reduce_epsilon = 0.000001  # 10^(-6)

        self.input_layer_size = NO_SENSORS * NO_SENSORS + 4 + 2   # daca e obstacol sus, jos, stanga, dreapta
        self.batch_size = 0
        self.epochs = 0

        self.model_params = {}

    def get_direction(self):
        current_state = self.p.getGameState()

        if self.previous_state.get('snake_head_x') < current_state.get('snake_head_x') and \
                self.previous_state.get('snake_head_y') == current_state.get('snake_head_y'):
            self.directie = 'dreapta'
        elif self.previous_state.get('snake_head_x') > current_state.get('snake_head_x') and \
                self.previous_state.get('snake_head_y') == current_state.get('snake_head_y'):
            self.directie = 'stanga'
        elif self.previous_state.get('snake_head_x') == current_state.get('snake_head_x') and \
                self.previous_state.get('snake_head_y') > current_state.get('snake_head_y'):
            self.directie = 'sus'
        elif self.previous_state.get('snake_head_x') == current_state.get('snake_head_x') and \
                self.previous_state.get('snake_head_y') < current_state.get('snake_head_y'):
            self.directie = 'jos'

    def get_sensors(self):
        screen_rgb = self.p.getScreenRGB()
        screen = self.game.getScreen()

        x_snake_head, y_snake_head = self.game.getActualHeadPosition(self.directie)

        pygame.draw.circle(screen, (255, 255, 255), (x_snake_head, y_snake_head), 2)
        pygame.display.update()

        no_rows_matrix = NO_SENSORS
        no_columns_matrix = NO_SENSORS

        print_value = 20

        food_color = self.game.getFoodColor()
        snake_color = self.game.getSnakeColor()

        # up
        if self.directie == "sus":
            sensors = np.zeros((no_columns_matrix, no_rows_matrix))
            row = 0
            for pos_x in range(x_snake_head - no_columns_matrix // 2,
                               x_snake_head + np.ceil(no_columns_matrix / 2).astype(int)):
                col = 0
                for pos_y in range(y_snake_head - no_rows_matrix, y_snake_head):
                    if (0 <= pos_x < HEIGHT) and (0 <= pos_y < WIDTH):
                        # set sensors values
                        if np.array_equal(screen_rgb[pos_x][pos_y], np.array(food_color)):
                            sensors[row][col] = 1
                        elif np.array_equal(screen_rgb[pos_x][pos_y], np.array(snake_color)):
                            sensors[row][col] = -1

                        # show sensors
                        if (pos_x - x_snake_head) % print_value == 0 and (pos_y - y_snake_head) % print_value == 0:
                            pygame.draw.circle(screen, (255, 255, 255), (pos_x, pos_y), 2)
                            pygame.display.flip()
                    else:
                        sensors[row][col] = -1
                    col += 1
                row += 1
            return np.rot90(sensors, -1)
            # return sensors.T

        # right
        if self.directie == "dreapta":
            sensors = np.zeros((no_rows_matrix, no_columns_matrix))
            row = 0
            for pos_x in range(x_snake_head, x_snake_head + no_rows_matrix):
                col = 0
                for pos_y in range(y_snake_head - no_columns_matrix // 2,
                                   y_snake_head + np.ceil(no_columns_matrix / 2).astype(int)):
                    if (0 <= pos_x < HEIGHT) and (0 <= pos_y < WIDTH):
                        # set sensors values
                        if np.array_equal(screen_rgb[pos_x][pos_y], np.array(food_color)):
                            sensors[row][col] = 1
                        elif np.array_equal(screen_rgb[pos_x][pos_y], np.array(snake_color)):
                            sensors[row][col] = -1

                        # show sensors
                        if (pos_x - x_snake_head) % print_value == 0 and (pos_y - y_snake_head) % print_value == 0:
                            pygame.draw.circle(screen, (255, 255, 255), (pos_x, pos_y), 2)
                            pygame.display.flip()
                    else:
                        sensors[row][col] = -1
                    col += 1
                row += 1
            return np.rot90(sensors, 2)
            # return sensors.T

        # down
        if self.directie == "jos":
            sensors = np.zeros((no_columns_matrix, no_rows_matrix))
            row = 0
            for pos_x in range(x_snake_head - no_columns_matrix // 2,
                               x_snake_head + np.ceil(no_columns_matrix / 2).astype(int)):
                col = 0
                for pos_y in range(y_snake_head, y_snake_head + no_rows_matrix):
                    if (0 <= pos_x < HEIGHT) and (0 <= pos_y < WIDTH):
                        # set sensors values
                        if np.array_equal(screen_rgb[pos_x][pos_y], np.array(food_color)):
                            sensors[row][col] = 1
                        elif np.array_equal(screen_rgb[pos_x][pos_y], np.array(snake_color)):
                            sensors[row][col] = -1

                        # show sensors
                        if (pos_x - x_snake_head) % print_value == 0 and (pos_y - y_snake_head) % print_value == 0:
                            pygame.draw.circle(screen, (255, 255, 255), (pos_x, pos_y), 2)
                            pygame.display.flip()
                    else:
                        sensors[row][col] = -1
                    col += 1
                row += 1
            return np.rot90(sensors, 1)
            # return sensors.T

        # left
        if self.directie == "stanga":
            sensors = np.zeros((no_rows_matrix, no_columns_matrix))
            row = 0
            for pos_x in range(x_snake_head, x_snake_head - no_rows_matrix, -1):
                col = 0
                for pos_y in range(y_snake_head - no_columns_matrix // 2,
                                   y_snake_head + np.ceil(no_columns_matrix / 2).astype(int)):
                    if (0 <= pos_x < HEIGHT) and (0 <= pos_y < WIDTH):
                        # set sensors values
                        if np.array_equal(screen_rgb[pos_x][pos_y], np.array(food_color)):
                            sensors[row][col] = 1
                        elif np.array_equal(screen_rgb[pos_x][pos_y], np.array(snake_color)):
                            sensors[row][col] = -1

                        # show sensors
                        if (pos_x - x_snake_head) % print_value == 0 and (pos_y - y_snake_head) % print_value == 0:
                            pygame.draw.circle(screen, (255, 255, 255), (pos_x, pos_y), 2)
                            pygame.display.flip()
                    else:
                        sensors[row][col] = -1
                    col += 1
                row += 1
            return np.rot90(sensors, 2)
            # return np.rot90(sensors, 3)

    def get_sensors_prime(self):
        no_rows_matrix = NO_SENSORS
        no_columns_matrix = NO_SENSORS
        sensors = np.zeros((no_rows_matrix, no_columns_matrix))
        x_snake_head = int(self.p.getGameState().get('snake_head_x'))
        y_snake_head = int(self.p.getGameState().get('snake_head_y'))
        x_food = int(self.p.getGameState().get('food_x'))
        y_food = int(self.p.getGameState().get('food_y'))
        if x_snake_head > no_rows_matrix:
            x_snake_head = no_rows_matrix - 1
        if y_snake_head > no_columns_matrix:
            y_snake_head = no_columns_matrix - 1
        if x_snake_head < 0:
            x_snake_head = 0
        if y_snake_head < 0:
            y_snake_head = 0
        print(x_snake_head, y_snake_head)
        sensors[x_snake_head][y_snake_head] = 1
        sensors[x_food][y_food] = 7
        return sensors

    def get_dist_food_head(self):
        self.actions = self.p.getActionSet()
        current_state = self.p.getGameState()

        x_snake = current_state.get('snake_head_x')
        y_snake = current_state.get('snake_head_y')
        x_food = current_state.get('food_x')
        y_food = current_state.get('food_y')

        poz = [[0, -1], [-1, 0], [1, 0], [0, 1]]
        dists = []
        for count, action in enumerate(self.actions):
            if action:
                x = x_snake + poz[count][0]
                y = y_snake + poz[count][1]
                dists.append(self.euclidian_distance(x, y, x_food, y_food)[0])

        # setam la valoarea maxima directia imposibila
        dists[my_directions[oposite[self.directie]]] = 400

        return dists

    def get_dist_wall_head(self):
        current_state = self.p.getGameState()
        x_snake = current_state.get('snake_head_x')
        y_snake = current_state.get('snake_head_y')

        return [y_snake, x_snake, 400 - y_snake, 400 - x_snake]

    def get_action_intelligent(self):
        dists = self.get_dist_food_head()

        value = 3
        min_dist = 9999
        for count, i in enumerate(dists):
            if i < min_dist:
                value = count
                min_dist = i
        return value

    @staticmethod
    def euclidian_distance(ax, ay, bx, by):
        return np.around(np.array([math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)]), decimals=1)

    @staticmethod
    def manhattan_distance(ax, ay, bx, by):
        return np.around(np.array([abs(ax - bx) + abs(ay - by)]), decimals=1)

    def get_current_state(self):
        snake_state = self.p.getGameState()

        if len(self.previous_state) == 0:
            self.previous_state = snake_state
            sensors = self.get_sensors()
        else:
            self.get_direction()
            self.previous_state = snake_state
            sensors = self.get_sensors()

        current_state = sensors.reshape((NO_SENSORS * NO_SENSORS,))

        directions = {
            "sus": 0,
            "stanga": 0,
            "dreapta": 0,
            "jos": 0,
        }
        wrong_way = -1
        right_way = 1
        head_x = snake_state['snake_head_x']
        head_y = snake_state['snake_head_y']
        food_x = snake_state['food_x']
        food_y = snake_state['food_y']

        if head_x <= food_x and head_y <= food_y:
            directions["sus"] = wrong_way
            directions["stanga"] = wrong_way
            directions["dreapta"] = 1
            directions["jos"] = 1
        elif head_x <= food_x and head_y >= food_y:
            directions["sus"] = 1
            directions["stanga"] = wrong_way
            directions["dreapta"] = 1
            directions["jos"] = wrong_way
        elif head_x >= food_x and head_y >= food_y:
            directions["sus"] = 1
            directions["stanga"] = 1
            directions["dreapta"] = wrong_way
            directions["jos"] = wrong_way
        elif head_x >= food_x and head_y <= food_y:
            directions["sus"] = wrong_way
            directions["stanga"] = 1
            directions["dreapta"] = wrong_way
            directions["jos"] = 1

        directions = np.array([directions[key] for key in directions])
        # current_state = np.append(current_state, directions)

        distance = self.euclidian_distance(head_x, head_y, food_x, food_y)
        # current_state = np.append(current_state, distance)

        distance_s = np.array(self.get_dist_wall_head())
        distance_s = [-1 if i < 20 else i for i in distance_s]

        current_state = np.append(current_state, [distance_s])

        # pozitionarea realtiva sarpelui (in ce diretie merge)
        snake_direction = my_directions_coord[self.directie]
        current_state = np.append(current_state, snake_direction)

        # current_state = np.append(current_state, [food_x, food_y])

        # current_state = np.append(current_state, self.get_dist_food_head())

        current_state = current_state.reshape((1, self.input_layer_size))

        # print(current_state)
        return current_state

    def pick_action(self, current_state, mode):
        if mode == 'random':
            value = np.random.randint(0, len(self.actions))
            return self.actions[value]
        elif mode == 'fit':
            qval = self.model.predict(current_state)
            value = np.argmax(qval)
            return self.actions[value]
        elif mode == 'intelligent':
            value = self.get_action_intelligent()
            return self.actions[value]

    def build_model(self, file_saved_weights=''):
        model = Sequential()

        model.add(layers.Dense(self.model_params['dimension_layer1'],
                               activation=self.model_params['activation_layer1'],
                               kernel_initializer='glorot_normal',
                               input_shape=(self.input_layer_size,)))  # lecun_uniform #ceva random
        model.add(layers.Dense(self.model_params['dimension_layer2'],
                               activation=self.model_params['activation_layer2'],
                               kernel_initializer='glorot_normal'))

        model.add(layers.Dense(4, activation=self.model_params['activation_layer3']))

        model.compile(optimizer=optimizers.Adam(lr=1e-2), loss='mean_squared_error', metrics=['accuracy'])

        if file_saved_weights != '':
            model.load_weights(file_saved_weights)
        return model

    def closer_to_food(self, current_state, next_state):
        distance_current_state = self.euclidian_distance(current_state['snake_head_x'], current_state['snake_head_y'], current_state['food_x'], current_state['food_y'])
        distance_next_state = self.euclidian_distance(next_state['snake_head_x'], next_state['snake_head_y'], next_state['food_x'], next_state['food_y'])
        return distance_current_state >= distance_next_state

    def train_net(self, train_frames=0, batch_size=0, model_params=None, no_frames_to_save=0, ep=0, no_frames_between_trains=0):
        if train_frames != 0:
            self.train_frames = train_frames
        if batch_size != 0:
            self.batch_size = batch_size
        if model_params is not None:
            self.model_params = model_params
        if no_frames_to_save != 0:
            self.no_frames_to_save = no_frames_to_save
        if ep != 0:
            self.epochs = ep
        if no_frames_between_trains != 0:
            self.no_frames_between_trains = no_frames_between_trains

        self.model = self.build_model()

        replay = deque(maxlen=self.observe)
        current_state = self.get_current_state()

        no_frame = 0
        for no_frame in range(self.train_frames):
            # while True:
            if self.p.game_over():
                self.p.reset_game()

            if random.random() < self.epsilon or no_frame < self.observe:
                if random.random() < 0.1:
                    action = self.pick_action(current_state, mode='random')
                else:
                    action = self.pick_action(current_state, mode='random')
            else:
                action = self.pick_action(current_state, mode='fit')

            old_simple_game_state = self.p.getGameState()
            reward = self.p.act(action)

            new_simple_game_state = self.p.getGameState()
            new_state = self.get_current_state()

            numar_suficient_de_mic = 0.1
            if self.rewards['loss'] - numar_suficient_de_mic >= reward or reward >= self.rewards['loss'] + numar_suficient_de_mic:
                if self.closer_to_food(old_simple_game_state,  new_simple_game_state):
                    reward += self.rewards['close']

            if self.p.score() == self.rewards['positive']:
                # current_state += 1 # chestia asta adauga un 1 la toate chestiile din starea noastra, nu prea are sens
                reward += self.rewards['positive']

            if no_frame % 100 == 0:
                print("^^^^^^^^^^^^^^^^^^  STATES IN FRAME ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                print(old_simple_game_state)
                print(current_state)
                print(new_state)
                # if no_frame % 100 == 0:
                print("Frame number: " + str(no_frame))
                print("Current score: " + str(reward))

            if len(replay) == self.observe:
                replay.popleft()

            replay.append((current_state, action, reward, new_state))

            if no_frame > self.observe and no_frame % self.no_frames_between_trains == 0:
                minibatch = random.sample(replay, self.batch_size)
                X_train, Y_train = self.process_minibatch(minibatch)

                self.model.fit(X_train, Y_train, epochs=self.epochs)  # batch_size #epochs

            current_state = new_state

            if self.epsilon > 0.1 and no_frame > self.observe:
                self.epsilon -= self.reduce_epsilon

            if (no_frame != 0) and (no_frame % self.no_frames_to_save == 0):
                self.save_model()
            no_frame += 1
        self.save_model()

    def process_minibatch(self, minibatch):
        len_minibatch = len(minibatch)

        old_states_replay = np.zeros((len_minibatch, self.input_layer_size))
        actions_replay = np.zeros((len_minibatch,))
        rewards_replay = np.zeros(len_minibatch, )
        new_states_replay = np.zeros((len_minibatch, self.input_layer_size))

        for index, memory in enumerate(minibatch):
            old_state_mem, action_mem, reward_mem, new_state_mem = memory
            old_states_replay[index] = old_state_mem.reshape(self.input_layer_size, )
            if action_mem == 119:  # up
                actions_replay[index] = 0
            elif action_mem == 97:  # left
                actions_replay[index] = 1
            elif action_mem == 100:  # right
                actions_replay[index] = 2
            elif action_mem == 115:  # down
                actions_replay[index] = 3
            rewards_replay[index] = reward_mem
            new_states_replay[index] = new_state_mem.reshape(self.input_layer_size, )

        old_qvals = self.model.predict(old_states_replay)
        new_qvals = self.model.predict(new_states_replay)

        maxQs = np.max(new_qvals, axis=1)

        target = old_qvals
        terminal_states_index = np.where(np.logical_or(rewards_replay == self.rewards['loss'], rewards_replay == self.rewards['positive']))[0]
        non_terminal_states_index = np.where(np.logical_and(rewards_replay != self.rewards['loss'], rewards_replay != self.rewards['positive']))[0]

        target[terminal_states_index, actions_replay[terminal_states_index].astype(int)] = rewards_replay[
            terminal_states_index]
        target[non_terminal_states_index, actions_replay[non_terminal_states_index].astype(int)] \
            = rewards_replay[non_terminal_states_index] + self.gamma * maxQs[non_terminal_states_index]

        return old_states_replay, target

    def save_model(self):
        today = datetime.datetime.today()
        file_name = 'day_' + str(today.day - 17) + '_saved_' + str(today.hour) + '_' + str(
            today.minute) + '_dnq.h5'
        self.model.save_weights(file_name)
        print("Model", file_name, "salvat!", sep=" ")

        # subprocess.Popen(
        #      ["python", "playing_snake.py", file_name, json.dumps(self.rewards), json.dumps(self.model_params)],
        #      env=original_env)

    def play_game(self, file_saved_weights, model_params=None):
        if model_params is not None:
            self.model_params = model_params

        self.model = self.build_model(file_saved_weights)
        score = 0
        while not self.p.game_over():
            current_state = self.get_current_state()
            action = self.pick_action(current_state, mode='fit')
            reward = self.p.act(action)
            # if reward >= self.rewards['positive']:
            if reward > 0:
                score += 1
            score += reward
        print("Score obtained:", score)


########################################################################################################################

if __name__ == "__main__":
    interface(True)

    game = Snake(width=WIDTH, height=HEIGHT)

    rewards = {
        "positive": 100.0,
        "loss": -10.0,
        "tick": 0.000,
        "close": 0.0
    }

    p = PLE(game, fps=30, force_fps=False, display_screen=True, reward_values=rewards)  # frame_skip=6

    agent = DQNAgent(p, game, rewards)

    p.init()

    nn_params = {
        "dimension_layer1": 150,
        "activation_layer1": "relu",
        "dimension_layer2": 40,
        "activation_layer2": "relu",
        "activation_layer3": "linear"
    }

    agent.train_net(batch_size=128, train_frames=20_000, model_params=nn_params, no_frames_to_save=2_000, ep=1, no_frames_between_trains=100)

    # agent.play_game(file_saved_weights='evitare_margini_cu_directie.h5', model_params=nn_params)

    # batch size 64 128 # epoci 1

