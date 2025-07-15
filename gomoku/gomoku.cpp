/**
 * Created by choucisan on 2025/7/14 20:10
 */


#include "gomoku.h"
#include <iostream>
using namespace std;

constexpr int INPUT_SIZE = 18;
constexpr int OUTPUT_SIZE = 9;
constexpr float LEARNING_RATE =0.1;



gomoku::gomoku(int board_size, int win_length)
        : board_size(board_size), board(board_size * board_size, '.'), current_player(0), win_length(win_length) {}


void gomoku::init_game(GameState& state) {
    state.board = std::vector<char>(board_size * board_size, '.');
    state.current_player = 0;
}


void gomoku::display_board(GameState& state) {
    for (int row = 0; row < board_size; ++row) {
        for (int col = 0; col < board_size; ++col) {
            cout << state.board[row * board_size + col] << ' ';
        }
        cout << "   ";

        for (int col = 0; col < board_size; ++col) {
            int index = row * board_size + col;
            cout << index << ' ';
        }
        cout << '\n';
    }
    cout << std::endl;
}


void gomoku::board_to_inputs(GameState& state, vector<float>& inputs) {
    int total_cells = board_size * board_size;
    inputs.resize(total_cells * 2);

    for (int i = 0; i < total_cells; ++i) {
        if (state.board[i] == '.') {
            inputs[i * 2]     = 0.0f;
            inputs[i * 2 + 1] = 0.0f;
        } else if (state.board[i] == 'X') {
            inputs[i * 2]     = 1.0f;
            inputs[i * 2 + 1] = 0.0f;
        } else {  // 'O'
            inputs[i * 2]     = 0.0f;
            inputs[i * 2 + 1] = 1.0f;
        }
    }
}


char gomoku::check_game_over(GameState& state, char& winner) {
    // Check rows.
    for (int i = 0; i < 3; i++) {
        if (state.board[i*3] != '.' &&
            state.board[i*3] == state.board[i*3+1] &&
            state.board[i*3+1] == state.board[i*3+2]) {
            winner = state.board[i*3];
            return 1;
        }
    }

    // Check columns.
    for (int i = 0; i < 3; i++) {
        if (state.board[i] != '.' &&
            state.board[i] == state.board[i+3] &&
            state.board[i+3] == state.board[i+6]) {
            winner = state.board[i];
            return 1;
        }
    }

    // Check diagonals.
    if (state.board[0] != '.' &&
        state.board[0] == state.board[4] &&
        state.board[4] == state.board[8]) {
        winner = state.board[0];
        return 1;
    }
    if (state.board[2] != '.' &&
        state.board[2] == state.board[4] &&
        state.board[4] == state.board[6]) {
        winner = state.board[2];
        return 1;
    }

    // Check for tie (no free tiles left).
    int empty_tiles = 0;
    for (int i = 0; i < 9; i++) {
        if (state.board[i] == '.') empty_tiles++;
    }
    if (empty_tiles == 0) {
        winner = 'T';  // Tie
        return 1;
    }

    winner = '.'; // No winner yet
    return 0; // Game continues
}


int gomoku::get_computer_move(GameState& state, mlp& model, int display_probs) {
    vector<float> inputs(INPUT_SIZE);
    board_to_inputs(state, inputs);
    model.forward(inputs.data());

    float highest_prob = -1.0f;
    int highest_prob_idx = -1;
    int best_move = -1;
    float best_legal_prob = -1.0f;

    for (int i = 0; i < 9; ++i) {
        if (model.outputs[i] > highest_prob) {
            highest_prob = model.outputs[i];
            highest_prob_idx = i;
        }
        if (state.board[i] == '.' && (best_move == -1 || model.outputs[i] > best_legal_prob)) {
            best_move = i;
            best_legal_prob = model.outputs[i];
        }
    }

    if (display_probs) {
        cout << "Neural network move probabilities:\n";
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                int pos = row * 3 + col;
                printf("%5.1f%%", model.outputs[pos] * 100.0f);
                if (pos == highest_prob_idx) cout << "*";
                if (pos == best_move) cout << "#";
                cout << " ";
            }
            cout << "\n";
        }

        float total_prob = 0.0f;
        for (int i = 0; i < 9; ++i)
            total_prob += model.outputs[i];
        printf("Sum of all probabilities: %.2f\n\n", total_prob);
    }

    return best_move;
}

void gomoku::play_game(mlp& model) {
    GameState state;
    char winner;
    int move_history[9]; // Maximum 9 moves in a game.
    int num_moves = 0;

    init_game(state);

    cout << "Welcome to Tic Tac Toe! You are X, the computer is O.\n";
    cout << "Enter positions as numbers from 0 to 8 (see picture).\n";

    while (!check_game_over(state,winner)) {
        display_board(state);

        if (state.current_player == 0) {
            int move;
            char movec;
            std::cout << "Your move (0-8): ";
            std::cin >> movec;
            move = movec - '0';

            if (move < 0 || move > 8 || state.board[move] != '.') {
                std::cout << "Invalid move! Try again.\n";
                continue;
            }

            state.board[move] = 'X';
            move_history[num_moves++] = move;
        } else {
            std::cout << "Computer's move:\n";
            int move = get_computer_move(state,model,1);
            state.board[move] = 'O';
            std::cout << "Computer placed O at position " << move << "\n";
            move_history[num_moves++] = move;
        }

        state.current_player = !state.current_player;
    }

    display_board(state);

    if (winner == 'X') {
        std::cout << "You win!\n";
    } else if (winner == 'O') {
        std::cout << "Computer wins!\n";
    } else {
        std::cout << "It's a tie!\n";
    }

    learn_from_game(model, move_history, num_moves, 1, winner);
}


int gomoku::get_random_move(GameState& state) {
    while (true) {
        int move = rand() % 9;
        if (state.board[move] == '.') return move;
    }
}


void gomoku::learn_from_game(mlp &model, int *move_history, int num_moves, int nn_moves_even, char winner) {
    float reward;
    char nn_symbol = nn_moves_even ? 'O' : 'X';

    if (winner == 'T') reward = 0.3f;
    else if (winner == nn_symbol) reward = 1.0f;
    else reward = -2.0f;

    GameState state;
    std::vector<float> inputs(INPUT_SIZE);
    std::vector<float> target_probs(OUTPUT_SIZE);

    for (int move_idx = 0; move_idx < num_moves; move_idx++) {
        if ((nn_moves_even && move_idx % 2 != 1) ||
            (!nn_moves_even && move_idx % 2 != 0)) continue;

        init_game(state);

        for (int i = 0; i < move_idx; i++) {
            char symbol = (i % 2 == 0) ? 'X' : 'O';
            state.board[move_history[i]] = symbol;
        }

        board_to_inputs(state,inputs);
        model.forward(inputs.data());


        int move = move_history[move_idx];
        float move_importance = 0.5f + 0.5f * (float)move_idx / (float)num_moves;
        float scaled_reward = reward * move_importance;

        fill(target_probs.begin(), target_probs.end(), 0);

        if (scaled_reward >= 0) {
            target_probs[move] = 1;
        } else {
            int valid_moves_left = 0;
            for (int i = 0; i < 9; i++) {
                if (state.board[i] == '.' && i != move) valid_moves_left++;
            }
            if (valid_moves_left > 0) {
                float other_prob = 1.0f / valid_moves_left;
                for (int i = 0; i < 9; i++) {
                    if (state.board[i] == '.' && i != move) {
                        target_probs[i] = other_prob;
                    }
                }
            }
        }

        model.backward(target_probs, LEARNING_RATE, scaled_reward);
    }
}



char gomoku::play_random_game(mlp &model, int *move_history, int *num_moves) {
    GameState state;
    char winner = 0;
    *num_moves = 0;

    init_game(state);

    while (!check_game_over(state, winner)) {
        int move;

        if (state.current_player == 0) {  // Random player (X)
            move = get_random_move(state);
        } else {  // Use your mlp class instance
            move = get_computer_move(state,model,0);  // Pass your mlp instance
        }

        char symbol = (state.current_player == 0) ? 'X' : 'O';
        state.board[move] = symbol;
        move_history[(*num_moves)++] = move;

        state.current_player = !state.current_player;
    }

    // O (mlp) is side = 1
    learn_from_game(model, move_history, *num_moves, 1, winner);
    return winner;
}


