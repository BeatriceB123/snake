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

NO_SENSORS = 50


def interface(with_interface=False):
    if not with_interface:
        # os.putenv('SDL_VIDEODRIVER', "fbcon")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    #else:
        #os.putenv('SDL_VIDEODRIVER', "windib")
        #os.environ["SDL_VIDEODRIVER"] = None  # "windib" for windows ceva.. for linux
        #os.environ.setdefault()
        #del os.environ["SDL_VIDEODRIVER"]

#class interface():
#    def __init__(self,with_interface=False):
#        if not with_interface:
#            # os.putenv('SDL_VIDEODRIVER', "fbcon")
#            self.copy_env = os.environ.copy()
#            os.environ["SDL_VIDEODRIVER"] = "dummy"
#
#        #else:
#        #    os.putenv('SDL_VIDEODRIVER', "windib")
#            # os.environ["SDL_VIDEODRIVER"] = "directx"  # "windib" for windows ceva.. for linux
#            # os.environ.setdefault()
#    def __enter__(self):
#        return None
#    def __exit__(self, exc_type, exc_val, exc_tb):
#        os.environ = self.copy_env
#
#def play_game_cuurent_model(filename_weights, nn_params, rewards):
#
#    print(os.getenv("SDL_VIDEODRIVER"))
#    #interface(True)
#    #print(os.getenv("SDL_VIDEODRIVER"))
#
#    new_game = Snake(width=WIDTH, height=HEIGHT)
#
#    new_p = PLE(new_game, fps=30, force_fps=False, display_screen=True, reward_values=rewards)
#
#    new_agent = DQNAgent(new_p, new_game, rewards)
#
#    new_p.init()
#
#    new_agent.play_game(file_saved_weights=filename_weights, model_params=nn_params)
#
#    #interface(False)

class DQNAgent:

    def __init__(self, p, game, rewards):
        self.p = p
        self.game = game
        self.rewards = rewards

        self.directie = 'dreapta'
        self.previous_state = {}
        self.actions = p.getActionSet()

        self.model = None
        #self.model_play = None

        self.train_frames = 4000
        self.observe = 1000
        self.no_frames_to_save = 2000

        self.epsilon = 2  # exploration rate
        self.gamma = 0.9  # discount rate

        self.input_layer_size = NO_SENSORS * NO_SENSORS + 5
        self.batch_size = 64
        # self.learning_rate = 0.001?
        self.epochs = 1

        self.model_params = {
            "dimension_layer1": 100,
            "activation_layer1": "relu",
            "dimension_layer2": 10,
            "activation_layer2": "softmax",
            "activation_layer3": "linear"
        }

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
        if snake_state['snake_head_x'] <= snake_state['food_x'] and snake_state['snake_head_y'] <= snake_state[
            'food_y']:
            directions["sus"] = -1
            directions["stanga"] = -1
            directions["dreapta"] = 1
            directions["jos"] = 1
        elif snake_state['snake_head_x'] <= snake_state['food_x'] and snake_state['snake_head_y'] >= snake_state[
            'food_y']:
            directions["sus"] = 1
            directions["stanga"] = -1
            directions["dreapta"] = 1
            directions["jos"] = -1
        elif snake_state['snake_head_x'] >= snake_state['food_x'] and snake_state['snake_head_y'] >= snake_state[
            'food_y']:
            directions["sus"] = 1
            directions["stanga"] = 1
            directions["dreapta"] = -1
            directions["jos"] = -1
        elif snake_state['snake_head_x'] >= snake_state['food_x'] and snake_state['snake_head_y'] <= snake_state[
            'food_y']:
            directions["sus"] = -1
            directions["stanga"] = 1
            directions["dreapta"] = -1
            directions["jos"] = 1

        directions = np.array([directions[key] for key in directions])
        current_state = np.append(current_state, directions)

        distance = np.sqrt((snake_state['snake_head_x'] - snake_state['food_x']) ** 2 + (
                snake_state['snake_head_y'] - snake_state['food_y']) ** 2)

        distance = np.around(np.array([distance]), decimals=1)
        current_state = np.append(current_state, distance)

        current_state = current_state.reshape((1, self.input_layer_size))
        return current_state

    def pick_action(self, current_state, mode):
        if mode == 'random':
            value = np.random.randint(0, len(self.actions))
            return self.actions[value]
        elif mode == 'fit':
            qval = self.model.predict(current_state)
            value = np.argmax(qval)
            return self.actions[value]

    def build_model(self, file_saved_weights=''):
        model = Sequential()
        model.add(layers.Dense(self.model_params['dimension_layer1'], activation=self.model_params['activation_layer1'],
                               kernel_initializer='lecun_uniform', input_shape=(self.input_layer_size,)))
        model.add(layers.Dense(self.model_params['dimension_layer2'], activation=self.model_params['activation_layer2'],
                               kernel_initializer='lecun_uniform'))
        model.add(layers.Dense(4, activation=self.model_params['activation_layer3']))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        if file_saved_weights != '':
            model.load_weights(file_saved_weights)
        return model

    def closer_to_food(self, current_state, next_state):
        distance_current_state = (current_state['snake_head_x'] - current_state['food_x']) ** 2 + (
                current_state['snake_head_y'] - current_state['food_y']) ** 2
        distance_next_state = (next_state['snake_head_x'] - next_state['food_x']) ** 2 + (
                next_state['snake_head_y'] - next_state['food_y']) ** 2
        return distance_current_state >= distance_next_state

    def train_net(self, train_frames=0, batch_size=0, model_params=None, no_frames_to_save=0, ep=0):
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

        self.model = self.build_model()

        replay = deque(maxlen=self.observe)
        current_state = self.get_current_state()

        for no_frame in range(self.train_frames):
            if self.p.game_over():
                self.p.reset_game()

            print("Frame number " + str(no_frame))

            if random.random() < self.epsilon or no_frame < self.observe:
                action = self.pick_action(current_state, mode='random')
            else:
                action = self.pick_action(current_state, mode='fit')

            old_simple_game_state = self.p.getGameState()

            reward = self.p.act(action)

            new_simple_game_state = self.p.getGameState()
            new_state = self.get_current_state()

            if reward != self.rewards['loss']:
                if self.closer_to_food(old_simple_game_state, new_simple_game_state):  # reward pt ca s-a apropiat de mancare
                    reward += self.rewards['close']

            if len(replay) == self.observe:
                replay.popleft()


            replay.append((current_state, action, reward, new_state))

            if no_frame > self.observe:
                minibatch = random.sample(replay, self.batch_size)
                X_train, Y_train = self.process_minibatch(minibatch)

                self.model.fit(X_train, Y_train, epochs=self.epochs)  # batch_size #epochs

            current_state = new_state

            if self.epsilon > 0.1 and no_frame > self.observe:
                self.epsilon -= (1.0 / self.train_frames)

            if no_frame != 0 and no_frame % self.no_frames_to_save == 0:
                self.save_model()

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
        terminal_states_index = \
            np.where(np.logical_or(rewards_replay == self.rewards['loss'], rewards_replay == self.rewards['positive']))[
                0]
        non_terminal_states_index = \
            np.where(
                np.logical_and(rewards_replay != self.rewards['loss'], rewards_replay != self.rewards['positive']))[0]

        target[terminal_states_index, actions_replay[terminal_states_index].astype(int)] = rewards_replay[
            terminal_states_index]
        target[non_terminal_states_index, actions_replay[non_terminal_states_index].astype(int)] \
            = rewards_replay[non_terminal_states_index] + self.gamma * maxQs[non_terminal_states_index]

        return old_states_replay, target

    def save_model(self):
        today = datetime.datetime.today()
        file_name = 'day_' + str(today.day - 17) + '_saved_' + str(today.hour) + '_' + str(today.minute) + '_dnq.h5'
        self.model.save_weights(file_name)
        #print(os.getenv("SDL_VIDEODRIVER"))
        print("Model", file_name, "salvat!", sep=" ")

        #play_game_cuurent_model(file_name,self.model_params,self.rewards)

    def play_game(self, file_saved_weights, model_params=None):
        if model_params is not None:
            self.model_params = model_params

        self.model = self.build_model(file_saved_weights)
        #print("altceva")
        #print(os.getenv("SDL_VIDEODRIVER"))
        score = 0
        while not self.p.game_over():
            current_state = self.get_current_state()
            action = self.pick_action(current_state, mode='fit')
            reward = self.p.act(action)
            if reward == self.rewards['positive']:
                score += 1
        print("Score obtained:", score)



########################################################################################################################

if __name__ == "__main__":
    #print(os.getenv("SDL_VIDEODRIVER"))
    #interface(False)
    interface(True)

    game = Snake(width=WIDTH, height=HEIGHT)

    rewards = {
        "positive": 100.0,
        "loss": -70.0,
        "tick": -0.1,
        "close": 1.5
    }


    #with interface(False):
    p = PLE(game, fps=30, force_fps=False, display_screen=True, reward_values=rewards)

    #print(os.getenv("SDL_VIDEODRIVER"))

    agent = DQNAgent(p, game, rewards)

    p.init()

    nn_params = {
        "dimension_layer1": 200,
        "activation_layer1": "linear",
        "dimension_layer2": 30,
        "activation_layer2": "linear",
        "activation_layer3": "softmax"
    }
    #agent.train_net(train_frames=10000, batch_size=100, model_params=nn_params, no_frames_to_save=100, ep=6)

    agent.play_game(file_saved_weights='day_1_saved_9_15_dnq.h5', model_params=nn_params)

