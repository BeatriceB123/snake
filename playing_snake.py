import snake_project
from ple import PLE
from ple.games.snake import Snake

WIDTH = 400
HEIGHT = 400


def play_game_current_model(filename_weights, nn_params, rewards):
    new_game = Snake(width=WIDTH, height=HEIGHT)

    new_p = PLE(new_game, fps=30, force_fps=False, display_screen=True, reward_values=rewards)

    new_agent = snake_project.DQNAgent(new_p, new_game, rewards)

    new_p.init()

    new_agent.play_game(file_saved_weights=filename_weights, model_params=nn_params)


if __name__ == "__main__":

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

    play_game_current_model(filename_weights='day_1_saved_13_25_dnq.h5', nn_params=nn_params, rewards=rewards)
