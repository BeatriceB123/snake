import snake_project
from ple import PLE
from ple.games.snake import Snake
import sys
import json

WIDTH = 400
HEIGHT = 400

def play_game_current_model(filename_weights, nn_params, rewards):
    new_game = Snake(width=WIDTH, height=HEIGHT)

    new_p = PLE(new_game, fps=30, force_fps=False, display_screen=True, reward_values=rewards)

    new_agent = snake_project.DQNAgent(new_p, new_game, rewards)

    new_p.init()

    new_agent.play_game(file_saved_weights=filename_weights, model_params=nn_params)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        file = sys.argv[1]
        rewards = json.loads(sys.argv[2])
        nn_params = json.loads(sys.argv[3])

    else:
        file = 'day_1_saved_19_24_dnq.h5'
        rewards = {
            "positive": 100.0,
            "loss": -70.0,
            "tick": -0.1,
            "close": 1.5
        }

        nn_params = {
            "dimension_layer1": 200,
            "activation_layer1": "linear",
            "dimension_layer2": 30,
            "activation_layer2": "linear",
            "activation_layer3": "softmax"
        }

    play_game_current_model(filename_weights=file, nn_params=nn_params, rewards=rewards)
