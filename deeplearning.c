#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define LAST (NETWORK_SIZE + 1)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

typedef struct _LayerMem {
    float * input_matrix;
    float * weight_matrix;
    float * bias_matrix;
    float * output_matrix;
} Layer;

const int INPUT_SIZE = 784;
const int LAYER_SIZE = 16;
const int OUTPUT_SIZE = 10;
const int NETWORK_SIZE = 2;
const float HETA = 0.000085f;
const int EPOCHS = 50;

const char * train_file = "mnist_train.csv";
const char * test_file = "mnist_test.csv";

Layer * layers;
int input_label;
int output_label;
float * delta;
float * temp_delta;

FILE * readFile;

void allocateNetwork();
void deAllocateNetwork();
void populateNetwork();
void trainNetwork();
void testNetwork();

float rand_normal();
float random_weight();
void matrixMul(float * input, float * weights, float * biases, float * output, int n, int m, int applyActivation);
void inverseMatrixMul(float * delta, float * input, float * weights, float * output, int n, int m);
void calculateLabel();
int readline();
float activationFunction(float input);
float activationFunctionDerivative(float input);
void lastLayerActivationFunction(float *input, int size);
void backPropagate();

void printWeights();

int main(){
    allocateNetwork();
    populateNetwork();
    trainNetwork();
    testNetwork();
    deAllocateNetwork();
}

void printWeights(){
    printf("Last layer weights\n\n");
    for(int i = 0; i < OUTPUT_SIZE; i++){
        for(int j = 0; j < LAYER_SIZE; j++){
            printf("%.4f\t", layers[LAST].weight_matrix[i * LAYER_SIZE + j]);
        }
        printf("\n");
    }
    printf("3rd layer weights\n\n");
    for(int i = 0; i < LAYER_SIZE; i++){
        for(int j = 0; j < LAYER_SIZE; j++){
            printf("%.4f\t", layers[LAST - 1].weight_matrix[i * LAYER_SIZE + j]);
        }
        printf("\n");
    }
}

float rand_normal() {
    float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

float random_weight(int input_size){ // Use with Relu
    float stddev = sqrtf(2.0f / input_size);
    return rand_normal() * stddev;
}

void allocateNetwork(){
    layers = (Layer *)malloc(sizeof(Layer) * (NETWORK_SIZE + 2));
    
    layers[0].input_matrix = (float *)malloc(sizeof(float) * (INPUT_SIZE));
    layers[0].weight_matrix = NULL;
    layers[0].bias_matrix = NULL;
    layers[0].output_matrix = layers[0].input_matrix;

    layers[1].input_matrix = layers[0].output_matrix;
    layers[1].weight_matrix = (float *)malloc(sizeof(float) * (INPUT_SIZE * LAYER_SIZE));
    layers[1].bias_matrix = (float *)malloc(sizeof(float) * (LAYER_SIZE));
    layers[1].output_matrix = (float *)malloc(sizeof(float) * (LAYER_SIZE));

    for(int i = 2; i < LAST; i++){
        layers[i].input_matrix = layers[i - 1].output_matrix;
        layers[i].weight_matrix = (float *)malloc(sizeof(float) * (LAYER_SIZE * LAYER_SIZE));
        layers[i].bias_matrix = (float *)malloc(sizeof(float) * (LAYER_SIZE));
        layers[i].output_matrix = (float *)malloc(sizeof(float) * (LAYER_SIZE));
    }

    layers[LAST].input_matrix = layers[LAST - 1].output_matrix;
    layers[LAST].weight_matrix = (float *)malloc(sizeof(float) * (LAYER_SIZE * OUTPUT_SIZE));
    layers[LAST].bias_matrix = (float *)malloc(sizeof(float) * (OUTPUT_SIZE));
    layers[LAST].output_matrix = (float *)malloc(sizeof(float) * (OUTPUT_SIZE));

    delta = (float *)malloc(sizeof(float) * ((LAYER_SIZE > OUTPUT_SIZE)? LAYER_SIZE : OUTPUT_SIZE));
    temp_delta = (float *)malloc(sizeof(float) * ((LAYER_SIZE > OUTPUT_SIZE)? LAYER_SIZE : OUTPUT_SIZE));
}

void deAllocateNetwork(){
    for(int i = 0; i <= LAST; i++){
        if(layers[i].input_matrix) free(layers[i].input_matrix);
        if(layers[i].weight_matrix) free(layers[i].weight_matrix);
        if(layers[i].bias_matrix) free(layers[i].bias_matrix);
        if(layers[i].output_matrix) free(layers[i].output_matrix);
    }
    free(delta);
    free(layers);
}

void populateNetwork(){
    for(int i = 1; i <= LAST; i++){
        int n_in, n_out;

        if(i == 1){
            n_in = INPUT_SIZE;
            n_out = LAYER_SIZE;
        } else if(i == LAST){
            n_in = LAYER_SIZE;
            n_out = OUTPUT_SIZE;
        } else {
            n_in = LAYER_SIZE;
            n_out = LAYER_SIZE;
        }

        int count = n_in * n_out;
        for(int j = 0; j < count; j++){
            layers[i].weight_matrix[j] = random_weight(n_in);
        }
        for(int j = 0; j < n_out; j++){
            layers[i].bias_matrix[j] = 0.0f;
        }
    }
}

void matrixMul(float * input, float * weights, float * biases, float * output, int n, int m, int applyActivation){
    for(int line = 0; line < n; line++){
        output[line] = (biases == NULL)? 0.0f : biases[line];
        for(int column = 0; column < m; column++){
            output[line] += input[column] * weights[column + line * m];
        }
        if(applyActivation){
            output[line] = activationFunction(output[line]);
        }
    }
}

void inverseMatrixMul(float * delta, float * input, float * weights, float * output, int n, int m){
    for(int column = 0; column < m; column++){
        output[column] = 0.0f;
        for(int line = 0; line < n; line++){
            output[column] += delta[line] * weights[column + line * m];
        }
        output[column] *= activationFunctionDerivative(input[column]);
    }
}

void calculateLabel(){
    for(int i = 1; i <= LAST; i++){
        int n_in, n_out;

        if(i == 1){
            n_in = INPUT_SIZE;
            n_out = LAYER_SIZE;
        } else if(i == LAST){
            n_in = LAYER_SIZE;
            n_out = OUTPUT_SIZE;
        } else {
            n_in = LAYER_SIZE;
            n_out = LAYER_SIZE;
        }

        matrixMul(layers[i].input_matrix, layers[i].weight_matrix, layers[i].bias_matrix, layers[i].output_matrix, n_out, n_in, i != LAST);
    }

    lastLayerActivationFunction(layers[LAST].output_matrix, OUTPUT_SIZE);
    output_label = 0;
    for(int i = 1; i < OUTPUT_SIZE; i++){
        if(layers[LAST].output_matrix[output_label] < layers[LAST].output_matrix[i]){
            output_label = i;
        }
    }
}

int readline(){
    char line[5000]; // large enough for one line
    if (!fgets(line, sizeof(line), readFile)) {
        return 0; // no more lines or error
    }

    // Extract the label (first token)
    char *token = strtok(line, ",");
    if (!token) return 0;
    input_label = atoi(token);

    // Extract the next 784 pixels
    for (int i = 0; i < INPUT_SIZE; i++) {
        token = strtok(NULL, ",");
        if (!token) return 0; // malformed line
        layers[0].input_matrix[i] = strtof(token, NULL) / 255.0f;
    }
    return 1; // success
}


float activationFunction(float input){
    if(isinf(input)){
        printf("Inf clipping has occured at activationFunction, you might want to reduce HETA\n");
        if(input > 0) return 1e+20f;
        else return -1e+20f;
    }
    else if(input > 0) return input;
    else return 0;
}

float activationFunctionDerivative(float input){
    if(input > 0) return 1;
    else return 0;
}

void lastLayerActivationFunction(float *input, int size) {
    for (int i = 0; i < size; i++) {
        if (isinf(input[i])){
            printf("Inf clipping has occured at lastLayerActivationFunction, you might want to reduce HETA\n");
            if(input[i] > 0)
                input[i] = 1e+20f;
            else
                input[i] = -1e+20f;
        }
    }
    float max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) max = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);  // for numerical stability
        sum += input[i];
    }

    if(sum < 0.000000001f) sum = 0.000000001f;
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

void backPropagate(){
    for(int i = 0; i < OUTPUT_SIZE; i++){
        delta[i] = (layers[LAST].output_matrix[i] - ((i == input_label)? 1.0f : 0.0f));
    }

    int n_in, n_out;


    for(int layer = LAST; layer > 0; layer--){
        if(layer == 1){
            n_in = INPUT_SIZE;
            n_out = LAYER_SIZE;
        } else if(layer == LAST){
            n_in = LAYER_SIZE;
            n_out = OUTPUT_SIZE;
        } else {
            n_in = LAYER_SIZE;
            n_out = LAYER_SIZE;
        }

        for(int line = 0; line < n_out; line++){
            for(int column = 0; column < n_in; column++){
                layers[layer].weight_matrix[line * n_in + column] -= delta[line] *
                HETA *
                layers[layer].input_matrix[column];
            }

            layers[layer].bias_matrix[line] -= delta[line] * HETA;
        }

        if(layer > 1){
            inverseMatrixMul(delta, layers[layer].input_matrix, layers[layer].weight_matrix, temp_delta, n_out, n_in);
            float * swapper = delta;
            delta = temp_delta;
            temp_delta = swapper;
        }
    }
}

void trainNetwork(){
    readFile = fopen(train_file, "r");
    for(int epoch = 0; epoch < EPOCHS; epoch++){
        if (!readFile) {
            perror("Failed to open train file");
            return;
        }

        float total_loss = 0.0f;
        int count = 0;

        while(readline()){
            calculateLabel();
            float prob = layers[LAST].output_matrix[input_label];
            if (prob < 0.000000001f) prob = 0.000000001f;  // avoid log(0) or log(negative)
            float loss = -logf(prob);
            total_loss += loss;
            count++;
            backPropagate();
        }
        printf("Epoch %d, Total Loss = %f, Count = %d, Average Loss = %.5f\n", epoch, total_loss, count, total_loss / count);
        rewind(readFile);
    }
    fclose(readFile);
}

void testNetwork(){
    readFile = fopen(test_file, "r");
    if (!readFile) {
        perror("Failed to open test file");
        return;
    }

    int correct = 0;
    int wrong = 0;
    while(readline()){
        calculateLabel();
        if(input_label != output_label){
            wrong++;
        }
        else{
            correct++;
        }
    }

    printf("Correct: %d\nWrong: %d\nTotal: %d\nAccuracy: %f", correct, wrong, correct + wrong, (float) correct / (correct + wrong));

    fclose(readFile);
}