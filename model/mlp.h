/**
 * Created by choucisan on 2025/7/14 19:07
 */


#ifndef GOMOKU_MLP_H
#define GOMOKU_MLP_H

#include <vector>



class mlp {
    public:
        mlp(int input_size, int hidden_size, int output_size, float learning_rate);

        void forward(const float* input);
        void backward(const std::vector<float>& target_probs, float learning_rate, float reward_scaling);
        void softmax(const std::vector<float>& input, std::vector<float>& output);
        float relu(float x);
        float relu_derivative(float x);

        std::vector<float> inputs;
        std::vector<float> hidden;
        std::vector<float> raw_logits;
        std::vector<float> outputs;

    private:
        int input_size, hidden_size, output_size;
        float learning_rate;

        std::vector<float> weights_ih;
        std::vector<float> weights_ho;
        std::vector<float> biases_h;
        std::vector<float> biases_o;


};

#endif // GOMOKU_MLP_H
