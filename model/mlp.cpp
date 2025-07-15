/**
 * Created by choucisan on 2025/7/14 19:07
 */


#include "mlp.h"
#include <cstdlib>
#include <ctime>

using namespace std;

mlp::mlp(int input_size, int hidden_size, int output_size, float learning_rate)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate),
          weights_ih(hidden_size * input_size), weights_ho(output_size * hidden_size),
          biases_h(hidden_size), biases_o(output_size),
          inputs(input_size), hidden(hidden_size), raw_logits(output_size), outputs(output_size)
{
    srand(static_cast<unsigned>(time(nullptr)));

    for (auto& w : weights_ih) w = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (auto& w : weights_ho) w = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    fill(biases_h.begin(), biases_h.end(), 0.0f);
    fill(biases_o.begin(), biases_o.end(), 0.0f);
}


float mlp::relu(float x) {
    return x > 0.0f ? x : 0.0f;
}


float mlp::relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}


void mlp::softmax(const std::vector<float>& input, std::vector<float>& output) {
    const size_t size = input.size();
    if (output.size() != size) {
        output.resize(size);
    }

    float max_val = *std::max_element(input.begin(), input.end());

    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    if (sum > 0.0f && std::isfinite(sum)) {
        for (size_t i = 0; i < size; ++i) {
            output[i] /= sum;
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            output[i] = 1.0f / static_cast<float>(size);
        }
    }
}

void mlp::forward(const float* input_array) {
    std::copy(input_array, input_array + input_size, inputs.begin());

    // Compute hidden layer activations
    for (int i = 0; i < hidden_size; ++i) {
        float sum = biases_h[i];
        for (int j = 0; j < input_size; ++j) {
            sum += inputs[j] * weights_ih[j * hidden_size + i];
        }
        hidden[i] = relu(sum);
    }

    // Compute output layer logits
    for (int i = 0; i < output_size; ++i) {
        float sum = biases_o[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden[j] * weights_ho[j * output_size + i];
        }
        raw_logits[i] = sum;
    }

    softmax(raw_logits, outputs);
}

void mlp::backward(const std::vector<float>& target_probs, float learning_rate, float reward_scaling) {
    std::vector<float> output_deltas(output_size);
    std::vector<float> hidden_deltas(hidden_size);


    // Output layer deltas (softmax + cross-entropy)
    for (int i = 0; i < output_size; ++i) {
        output_deltas[i] = (outputs[i] - target_probs[i]) * std::fabs(reward_scaling);
    }

    // Hidden layer deltas (backpropagate error through weights_ho and relu derivative)
    for (int i = 0; i < hidden_size; ++i) {
        float error = 0.0f;
        for (int j = 0; j < output_size; ++j) {
            error += output_deltas[j] * weights_ho[i * output_size + j];
        }
        hidden_deltas[i] = error * relu_derivative(hidden[i]);
    }


    // Update weights_ho and biases_o
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            weights_ho[i * output_size + j] -= learning_rate * output_deltas[j] * hidden[i];
        }
    }
    for (int j = 0; j < output_size; ++j) {
        biases_o[j] -= learning_rate * output_deltas[j];
    }

    // Update weights_ih and biases_h
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            weights_ih[i * hidden_size + j] -= learning_rate * hidden_deltas[j] * inputs[i];
        }
    }
    for (int j = 0; j < hidden_size; ++j) {
        biases_h[j] -= learning_rate * hidden_deltas[j];
    }
}