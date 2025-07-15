/**
 * Created by choucisan on 2025/7/14 21:25
 */


#include <iostream>
#include <cstdlib>
#include <ctime>
#include "gomoku/gomoku.h"
#include "model/mlp.h"

using namespace std;



void train_against_random(mlp &model, int num_games) {
    int move_history[9];
    int num_moves;
    int wins = 0, losses = 0, ties = 0;

    cout << "Training neural network against " << num_games << " random games...\n";
    gomoku game(3, 3);

    int played_games = 0;
    for (int i = 0; i < num_games; ++i) {
        char winner = game.play_random_game(model, move_history, &num_moves);
        ++played_games;

        if (winner == 'O') {
            ++wins;
        } else if (winner == 'X') {
            ++losses;
        } else {
            ++ties;
        }

        if ((i + 1) % 10000 == 0) {
            float win_rate = (wins * 100.0f) / played_games;
            float loss_rate = (losses * 100.0f) / played_games;
            float tie_rate = (ties * 100.0f) / played_games;

            cout << "Games: " << (i + 1)
            << ", Wins: " << wins << " (" << win_rate << "%)"
            << ", Losses: " << losses << " (" << loss_rate << "%)"
            << ", Ties: " << ties << " (" << tie_rate << "%)\n";


            played_games = 0;
            wins = 0;
            losses = 0;
            ties = 0;
        }
    }

    cout << "\nTraining complete!\n";
}



int main(int argc, char** argv) {
    int random_games = 150000; // 默认对局次数

    if (argc > 1) {
        random_games = atoi(argv[1]);
    }

    srand(static_cast<unsigned int>(time(nullptr)));

    mlp model(18, 100, 9,0.1);


    if (random_games > 0) {
        train_against_random(model, random_games);
    }


    while (true) {
        gomoku game(3, 3);
        game.play_game(model);
        cout << "Play again? (y/n): ";
        char play_again;
        cin >> play_again;

        if (play_again != 'y' && play_again != 'Y') {
            break;
        }
    }

    return 0;
}