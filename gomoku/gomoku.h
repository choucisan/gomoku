/**
 * Created by choucisan on 2025/7/14 20:10
 */

#ifndef GOMOKU_GOMOKU_H
#define GOMOKU_GOMOKU_H

#include <vector>
#include <algorithm>
#include "../model/mlp.h"



class gomoku {struct GameState {
        std::vector<char> board;
        int current_player;

        GameState(int board_size = 3) : board(board_size * board_size, '.'), current_player(0) {}
    };

public:
    gomoku(int board_size, int win_length);

    void init_game(GameState& state);
    void display_board(GameState& state);
    void board_to_inputs(GameState& state,std::vector<float>& inputs);
    char check_game_over(GameState& state, char& winner);
    int get_computer_move(GameState& state,mlp& model,int display_probs);

    void learn_from_game(mlp& model, int* move_history, int num_moves, int nn_moves_even, char winner);
    void play_game(mlp& model);
    int get_random_move(GameState& state);
    char play_random_game(mlp& model, int* move_history, int* num_moves);

private:
    std::vector<char> board;
    int current_player;
    int board_size;
    int win_length;
};

#endif // GOMOKU_GOMOKU_H