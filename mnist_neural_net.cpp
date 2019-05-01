/*
 * Author: Adrian Alberto
 * Architect: Dr. Evangelos Yfantis
 */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <cmath>


int max_epoch = 16;
float learning_rate = 0.1f;
int outputPeriod = 6000; //number of iterations before outputting a partial accuracy


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

float input[28][28] = { 0 };
float w_ic[8][5][5] = { 0 };
float b_c[8] = { 0 };
float y_c[8][24][24] = { 0 };
float y_m[8][12][12] = { 0 };
float w_mh[8][12][12][45] = { 0 };
float b_h[45] = { 0 };
float y_h[45] = { 0 };
float w_ho[45][10] = { 0 };
float b_o[10] = { 0 };
float z_o[10] = { 0 };
float t[10] = { 0 };

float train_input[60000][28][28] = { 0 };
int train_label[60000] = { 0 };
float test_input[10000][28][28] = { 0 };
int test_label[10000] = { 0 };

void loadWeights(char*);
void initRandom();
int loadData();

int evaluate(); //returns 1 if correct, 0 if incorrect
void descend(float);

float max(float, float, float, float);
float activate(float);
float dActivate(float);

void doSingleTest(int);
int checkTestAccuracy();
void outputWeights(int, int);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

int main(int argc, char* argv[])
{
    if (loadData() == 1)
    {
        std::cout << "Make sure the four .ubyte files are in your working directory!\n";
        return 0;
    }

    initRandom();

    if (argc == 2)
    {
        std::cout << "Loading weights from " << argv[1] << std::endl;
        loadWeights(argv[1]);
        std::cout << "Running initial model test... ";
        std::cout << checkTestAccuracy() << "/10000 correct." << std::endl;

        int i = 0;
        while (true)
        {
            std::cout << "Enter an index 0-9999 to test the model, or -1 to resume training.\n";
            std::cout << "INPUT: ";
            std::cin >> i;
            if (i == -1)
                break;
            else if (i >= 0 && i <= 9999)
                doSingleTest(i);
        }
    }
    else
    { 
        std::cout << "INSTRUCTIONS\nIf you would like to load a model file, run the program with the filename for the epochX_correctXXXX.txt file.\n";
    }

    
    int startTime = (int)time(0);
    for (int epoch = 0; epoch < max_epoch; epoch++)
    {
        std::cout << "-----------------------------\nBeginning epoch..." << std::endl;
        int numCorrect = 0;
        int numCorrectPeriod = 0;
        for (int n = 0; n < 60000; n++)
        {
            if (n % outputPeriod == outputPeriod - 1)
            {
                int dt = (int)time(0) - startTime;
                int min = dt / 60;
                int sec = dt - min * 60;
                std::cout << "Progress: " << n + 1 << "/60000 ... ";
                std::cout << "partial accuracy: " << numCorrectPeriod * 100.0f / outputPeriod;
                std::cout << "% @ " << min << "m " << sec << "s" << std::endl;
                numCorrectPeriod = 0;
            }

            //Set inputs
            for (int x = 0; x < 28; x++)
                for (int y = 0; y < 28; y++)
                    input[x][y] = train_input[n][x][y];

            //Set truth values
            for (int i = 0; i < 10; i++)
            {
                t[i] = 0;
                if (i == train_label[n])
                    t[i] = 1;
            }

            //Feed-forward and record result
            if (train_label[n] == evaluate())
            {
                numCorrect++;
                numCorrectPeriod++;
            }

            //Back-propagate
            descend(learning_rate);
        }

        std::cout << "Testing..." << std::endl;
        int testCorrect = checkTestAccuracy();

        std::cout << "epoch " << epoch << ": " << ((float)numCorrect) / 600.0f << "% training accuracy; ";
        std::cout << ((float)testCorrect) / 100.0f << "% test accuracy" << std::endl;

        outputWeights(epoch, testCorrect);
    }


    return 0;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

void initRandom()
{
    srand((unsigned int)time(0));
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 5; j++)
            for (int k = 0; k < 5; k++)
                w_ic[i][j][k] = (float)rand() / (float)(RAND_MAX)-0.5f;

    for (int i = 0; i < 8; i++)
    {
        b_c[i] = (float)rand() / (float)(RAND_MAX)-0.5f;
        for (int j = 0; j < 12; j++)
            for (int k = 0; k < 12; k++)
                w_mh[i][j][k];
    }

    for (int i = 0; i < 45; i++)
    {
        b_h[i] = (float)rand() / (float)(RAND_MAX)-0.5f;
        for (int j = 0; j < 10; j++)
            w_ho[i][j] = (float)rand() / (float)(RAND_MAX)-0.5f;
    }
    for (int i = 0; i < 10; i++)
        b_o[i] = (float)rand() / (float)(RAND_MAX)-0.5f;
}

int loadData()
{
    std::cout << "Loading training/testing data... ";
    std::ifstream trainingFile("train-images-idx3-ubyte", std::ios::in | std::ios::binary);
    trainingFile.seekg(16);
    if (trainingFile.fail())
    {
        std::cout << "Failed to load training input data.\n";
        return 1;
    }


    std::ifstream labelFile("train-labels-idx1-ubyte", std::ios::in | std::ios::binary);
    labelFile.seekg(8);
    if (labelFile.fail())
    {
        std::cout << "Failed to load training input labels.\n";
        return 1;
    }

    unsigned char byte;

    for (int n = 0; n < 60000; n++) {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++) {
                trainingFile.read((char*)&byte, 1);
                train_input[n][i][j] = (float)(int)byte / 255.0f;
            }
        }
        labelFile.read((char*)&byte, 1);
        train_label[n] = (int)byte;
    }

    std::ifstream testingFile("t10k-images-idx3-ubyte", std::ios::in | std::ios::binary);
    testingFile.seekg(16);
    if (testingFile.fail())
    {
        std::cout << "Failed to load testing input data.\n";
        return 1;
    }

    std::ifstream testLabelFile("t10k-labels-idx1-ubyte", std::ios::in | std::ios::binary);
    testLabelFile.seekg(8);
    if (testLabelFile.fail())
    {
        std::cout << "Failed to load testing input labels.\n";
        return 1;
    }

    for (int n = 0; n < 10000; n++) {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++) {
                testingFile.read((char*)&byte, 1);
                test_input[n][i][j] = (float)(int)byte / 255.0f;
            }
        }
        testLabelFile.read((char*)&byte, 1);
        test_label[n] = (int)byte;
    }

    trainingFile.close();
    labelFile.close();
    testingFile.close();
    testLabelFile.close();

    std::cout << " done!\n";
    return 0;
}

int evaluate()
{
    //convolution layer
    for (int frame = 0; frame < 8; frame++)
    {
        for (int cx = 0; cx < 24; cx++)
        {
            for (int cy = 0; cy < 24; cy++)
            {
                float s = b_c[frame];
                for (int ix = 0; ix < 5; ix++)
                    for (int iy = 0; iy < 5; iy++)
                        s += (input[cx + ix][cy + iy] * w_ic[frame][ix][iy]);
                y_c[frame][cx][cy] = activate(s);
            }
        }
    }

    //max layer
    for (int frame = 0; frame < 8; frame++)
    {
        for (int mx = 0; mx < 12; mx++)
        {
            for (int my = 0; my < 12; my++)
            {
                y_m[frame][mx][my] = max(
                    y_c[frame][mx * 2][my * 2],
                    y_c[frame][mx * 2][my * 2 + 1],
                    y_c[frame][mx * 2 + 1][my * 2],
                    y_c[frame][mx * 2 + 1][my * 2 + 1]
                );
            }
        }
    }

    //hidden layer
    for (int i = 0; i < 45; i++)
    {
        float s = b_h[i];
        for (int frame = 0; frame < 8; frame++)
            for (int mx = 0; mx < 12; mx++)
                for (int my = 0; my < 12; my++)
                    s += y_m[frame][mx][my] * w_mh[frame][mx][my][i];
        y_h[i] = activate(s);
    }

    //output layer
    float max_output = -10.0f;
    int max_index = -1;
    for (int j = 0; j < 10; j++)
    {
        float s = b_o[j];
        for (int i = 0; i < 45; i++)
            s += y_h[i] * w_ho[i][j];
        z_o[j] = activate(s);
        if (z_o[j] > max_output)
        {
            max_output = z_o[j];
            max_index = j;
        }
    }

    return max_index;
}

void descend(float eta)
{
    float G0 = 0;
    float G_h[45] = { 0 };
    for (int j = 0; j < 10; j++)
    {
        float g0 = eta * (z_o[j] - t[j]) * dActivate(z_o[j]);
        b_o[j] -= g0;
        G0 += g0;

        for (int i = 0; i < 45; i++)
        {
            G_h[i] += w_ho[i][j] * g0;
            w_ho[i][j] -= eta * g0 * y_h[i];
        }

    }

    for (int i = 0; i < 45; i++)
    {
        /*
        dE[j]   dz_o[j] ds_o[j] dy_h[i]
        dz_o[j] ds_o[j] dy_h[i] db_h[i]
          '-------.-------'      1
               G_h[i]
        */
        b_h[i] -= G_h[i];
    }

    for (int frame = 0; frame < 8; frame++)
    {
        float G_c = 0; //total gradient at the conv layer

        for (int mx = 0; mx < 12; mx++)
        {
            for (int my = 0; my < 12; my++)
            {
                //Total gradient across the max layer
                float G_m = 0;
                for (int i = 0; i < 45; i++)
                {
                    G_m += w_mh[frame][mx][my][i] * dActivate(y_h[i]) *G_h[i];
                    w_mh[frame][mx][my][i] -= G_h[i] * y_m[frame][mx][my];
                }
                //G_m /= 144.0f;

                for (int cx = mx * 2; cx < mx * 2 + 2; cx++)
                {
                    for (int cy = my * 2; cy < my * 2 + 2; cy++)
                    {
                        //Backpropagate only through max value of conv layer
                        if (y_c[frame][cx][cy] == y_m[frame][mx][my])
                        {
                            //g_c is the gradient at this convolutional layer at (cx,cy)
                            float g_c = G_m * dActivate(y_c[frame][cx][cy]);
                            G_c += g_c;

                            for (int ix = cx; ix < cx + 5; ix++)
                                for (int iy = cy; iy < cy + 5; iy++)
                                    w_ic[frame][ix - cx][iy - cy] -= input[ix][iy] * g_c;

                        }
                    }
                }

            }
        }

        b_c[frame] -= G_c;
    }
}

void doSingleTest(int inputIndex)
{
    for (int x = 0; x < 28; x++)
    {
        for (int y = 0; y < 28; y++)
        {
            input[x][y] = test_input[inputIndex][x][y];
            if (input[x][y] > 0.9)
                std::cout << "#";
            else if (input[x][y] > 0.5)
                std::cout << "O";
            else if (input[x][y] > 0.3)
                std::cout << ":";
            else
                std::cout << " ";
        }
        std::cout << std::endl;
    }

    //Set truth values
    for (int i = 0; i < 10; i++)
    {
        t[i] = 0;
        if (i == test_label[inputIndex])
            t[i] = 1;
    }

    //Feed-forward
    int bestGuess = evaluate();

    std::cout << "Predicted: " << bestGuess << ", Actual: " << test_label[inputIndex] << std::endl;
}

int checkTestAccuracy()
{
    int numCorrect = 0;
    for (int n = 0; n < 10000; n++)
    {
        //Set inputs
        for (int x = 0; x < 28; x++)
            for (int y = 0; y < 28; y++)
                input[x][y] = test_input[n][x][y];

        //Set truth values
        for (int i = 0; i < 10; i++)
        {
            t[i] = 0;
            if (i == test_label[n])
                t[i] = 1;
        }

        //Feed-forward
        int best_z = evaluate();
        //Check if correct
        if (best_z == test_label[n])
            numCorrect++;
    }
    return numCorrect;
}

void outputWeights(int epoch, int testCorrect)
{
    std::ostringstream oss;
    oss << "epoch" << epoch << "_correct" << testCorrect << ".txt";
    std::cout << "Outputting weight file: " << oss.str() << std::endl;

    std::ofstream weightfile(oss.str().c_str());

    if (weightfile.is_open())
    {
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 5; j++)
                for (int k = 0; k < 5; k++)
                    weightfile << w_ic[i][j][k] << std::endl;

        for (int i = 0; i < 8; i++)
        {
            weightfile << b_c[i] << std::endl;
            for (int j = 0; j < 12; j++)
                for (int k = 0; k < 12; k++)
                    for (int l = 0; l < 45; l++)
                        weightfile << w_mh[i][j][k][l] << std::endl;
        }

        for (int i = 0; i < 45; i++)
        {
            weightfile << b_h[i] << std::endl;
            for (int j = 0; j < 10; j++)
                weightfile << w_ho[i][j] << std::endl;
        }
        for (int i = 0; i < 10; i++)
            weightfile << b_o[i] << std::endl;

        weightfile.close();
    }
}

void loadWeights(char* fname)
{
    std::ifstream weightfile(fname);
    if (weightfile.fail())
    {
        std::cout << "Cannot open weight file.\n";
        return;
    }

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 5; j++)
            for (int k = 0; k < 5; k++)
                weightfile >> w_ic[i][j][k];
    
    for (int i = 0; i < 8; i++)
    {
        weightfile >> b_c[i];
        for (int j = 0; j < 12; j++)
            for (int k = 0; k < 12; k++)
                for (int l = 0; l < 45; l++)
                    weightfile >> w_mh[i][j][k][l];
    }

    for (int i = 0; i < 45; i++)
    {
        weightfile >> b_h[i];
        for (int j = 0; j < 10; j++)
            weightfile >> w_ho[i][j];
    }
    for (int i = 0; i < 10; i++)
        weightfile >> b_o[i];

    weightfile.close();
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//sigmoid
float activate(float s)
{
    return 1.0f / (1.0f + exp(-s));
}
//delta sigmoid
float dActivate(float z)
{
    return z * (1.0f - z);
}

float max(float a, float b, float c, float d)
{
    float left = fmax(a, b);
    float right = fmax(c, d);
    return fmax(left, right);
}