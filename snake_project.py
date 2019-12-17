import numpy as np
from ple import PLE

import os
import datetime
from ple.games.snake import Snake
import pygame
from keras import Sequential, layers
import random
from collections import deque

WIDTH = 400
HEIGHT = 400

input_layer_size = 50 * 50


class NaiveAgent:

    def __init__(self, p):
        self.instanta = p
        self.directie = 'dreapta'
        self.previous_state = {}
        self.actions = p.getActionSet()

        self.model = None
        self.train_frames = 5000
        self.epsilon = 1
        self.observe = 1000
        self.gamma = 0.9
        self.batch_size = 128

    def getDirection(self):
        current_state = self.instanta.getGameState()

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

    @property
    def getSensors(self):
        screen_rgb = self.instanta.getScreenRGB()
        screen = game.getScreen()

        x_snake_head, y_snake_head = game.getActualHeadPosition(self.directie)

        pygame.draw.circle(screen, (255, 255, 255), (x_snake_head, y_snake_head), 2)
        pygame.display.update()

        no_rows_matrix = 50
        no_columns_matrix = 50

        print_value = 20

        food_color = game.getFoodColor()
        snake_color = game.getSnakeColor()

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
            return sensors.T

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
            return sensors.T

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
            return sensors.T

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
            return np.rot90(sensors, 3)

    def getCurrentState(self):
        snake_state = self.instanta.getGameState()
        if len(self.previous_state) == 0:
            self.previous_state = snake_state
            sensors = self.getSensors
        else:
            self.getDirection()
            self.previous_state = snake_state
            sensors = self.getSensors

        current_state = sensors.reshape((1, input_layer_size))
        return current_state

    def pickAction(self, current_state, mode):  # reward
        if mode == 'random':
            value = np.random.randint(0, len(self.actions))
            # print(value)
            return self.actions[value]
        elif mode == 'fit':
            qval = self.model.predict(current_state)
            value = np.argmax(qval)
            return self.actions[value]

    def build_model(self, file_weights=''):
        model = Sequential()
        from keras.activations import elu
        model.add(layers.Dense(100, activation='linear', kernel_initializer='lecun_uniform', input_shape=(input_layer_size,)))
        model.add(layers.Dense(100, activation='linear', kernel_initializer='lecun_uniform'))
        model.add(layers.Dense(4, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        print(file_weights)
        if file_weights != '':
            model.load_weights(file_weights)
        return model

    def save_model(self):
        today = datetime.datetime.today()
        file_name = 'day_' + str(today.day - 17) + '_saved_' + str(today.hour) + '_' + str(today.minute) + '_dnq.h5'
        self.model.save_weights(file_name, overwrite=True)
        print("Model salvat!")

    def train_net(self):
        self.model = self.build_model()

        replay = deque(maxlen=self.observe)
        current_state = self.getCurrentState()

        for no_frame in range(self.train_frames):
            if self.instanta.game_over():
                self.instanta.reset_game()

            if random.random() < self.epsilon or no_frame < self.observe:
                action = self.pickAction(current_state, mode='random')
            else:
                action = self.pickAction(current_state, mode='fit')

            print(self.instanta.getGameState())
            print(action)
            reward = self.instanta.act(action)
            print(self.instanta.getGameState())

            new_state = self.getCurrentState()
            print(np.array_equal(current_state, new_state))

            if len(replay) == self.observe:
                replay.popleft()

            replay.append((current_state, action, reward, new_state))
            print((current_state, action, reward, new_state))
            print(len(replay))

            if no_frame > self.observe:
                minibatch = random.sample(replay, self.batch_size)
                X_train, Y_train = self.process_minibatch(minibatch)

                self.model.fit(X_train, Y_train, batch_size=self.batch_size)  # epocile

            current_state = new_state

            if self.epsilon > 0.1 and no_frame > self.observe:
                self.epsilon -= (1.0 / self.train_frames)

        print(len(replay))
        self.save_model()

    def process_minibatch(self, minibatch):
        length = len(minibatch)

        old_states = np.zeros((length, input_layer_size))
        actions = np.zeros((length,))
        rewards_memory = np.zeros(length, )
        new_states = np.zeros((length, input_layer_size))

        for index, memory in enumerate(minibatch):
            old_state_mem, action_mem, reward_mem, new_state_mem = memory
            old_states[index] = old_state_mem.reshape(input_layer_size, )
            if action_mem == 119:  # up
                actions[index] = 0
            elif action_mem == 97:  # left
                actions[index] = 1
            elif action_mem == 100:  # right
                actions[index] = 2
            elif action_mem == 115:  # down
                actions[index] = 3
            rewards_memory[index] = reward_mem
            new_states[index] = new_state_mem.reshape(input_layer_size, )

        print(old_states.shape)
        print(actions.shape)
        print(rewards_memory.shape)
        print(new_states.shape)

        old_qvals = self.model.predict(old_states)
        new_qvals = self.model.predict(new_states)

        print(old_qvals.shape)
        print(new_qvals.shape)

        maxQs = np.max(new_qvals, axis=1)
        print(maxQs.shape)

        target = old_qvals
        terminal_states_index = \
            np.where(np.logical_or(rewards_memory == rewards['loss'], rewards_memory == rewards['positive']))[0]
        non_terminal_states_index = \
            np.where(np.logical_and(rewards_memory != rewards['loss'], rewards_memory != rewards['positive']))[0]

        print(terminal_states_index.shape)
        print(non_terminal_states_index.shape)

        target[terminal_states_index, actions[terminal_states_index].astype(int)] = rewards_memory[
            terminal_states_index]
        target[non_terminal_states_index, actions[non_terminal_states_index].astype(int)] \
            = rewards_memory[non_terminal_states_index] + self.gamma * maxQs[non_terminal_states_index]

        return old_states, target

    def play_game(self, file_weights):
        self.model = self.build_model(file_weights)

        score = 0
        while not self.instanta.game_over():
            current_state = self.getCurrentState()
            action = self.pickAction(current_state, mode='fit')
            reward = self.instanta.act(action)
            if reward == rewards['positive']:
                score += 1

        print("Score obtained:", score)


def interface(with_interface=False):
    if not with_interface:
        os.putenv('SDL_VIDEODRIVER', "fbcon")
        os.environ["SDL_VIDEODRIVER"] = "dummy"

########################################################################################################################


interface(True)

game = Snake(width=WIDTH, height=HEIGHT)

rewards = {
    "positive": 100.0,
    "loss": -70.0,
    "close": 1.4,
    "tick": -0.1
}

p = PLE(game, fps=30, force_fps=False, display_screen=True, reward_values=rewards)

agent = NaiveAgent(p)

p.init()

agent.train_net()

# agent.play_game(file_weights='day_0_saved_11_55_dnq.h5')

