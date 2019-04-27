/*
 * Adrian Alberto
 * 5002933684
 */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <bits/stdc++.h>
#include <sstream>

//Hyperparameters
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int max_epoch = 8;
float learning_rate = 0.2;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

float input[28][28] = {0};    //input
float w_ic[6][5][5] = {0};    //input->convolution weights
float b_c[6][24][24] = {0};   //convolution bias
float y_c[6][24][24] = {0};   //convolution layer activation output
float w_cm[6][12][12] = {0};  //conv->max layer weights
float y_m[6][12][12] = {0};   //max layer activation output
float u[6][12][12][45] = {0}; //max->sigmoid layer weights
float b_sig[45] = {0};        //sigmoid layer bias
float y_sig[45] = {0};        //sigmoid layer activation output
float v[45][10] = {0};        //sigmoid->output layer weights
float b_out[10] = {0};        //output layer bias
float z_out[10] = {0};        //output
float t[10] = {0};            //truth values
float E[10] = {0};            //error per output node

float train[60000][28][28] = {0}; //training inputs
float train_label[60000] = {0};   //training outputs
float test[10000][28][28] = {0};
float test_label[10000] = {0};

void init();
void loadData();
void evaluate();
void descend(float);
float max(float, float, float, float);
int checkTestAccuracy();
void outputWeights(int, int);

int main()
{
    //Load weights and biases from file
    init();

    //Load training data
    loadData();
    std::cout << "Testing model...\n";
    int testCorrect = checkTestAccuracy();
    std::cout << "\ntest accuracy: " << ((float)testCorrect*100.0f/10000.0f) << "%" << std::endl;

    int inputIndex = 0;
    while (inputIndex >= 0)
    {
        std::cout << "Enter an index for the test file to test (0-9999) or -1 to exit: ";
        std::cin >> inputIndex;
        if (inputIndex < 0 || inputIndex > 9999)
            break;

        //Set inputs
        for (int x = 0; x < 28; x++)
        {
            for (int y = 0; y < 28; y++)
            {
                input[x][y] = test[inputIndex][x][y];
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
        evaluate();

        //Check if correct
        float max = 0;
        int best_z = 0;
        for (int i = 0; i < 10; i++)
        {
            if (z_out[i] > max)
            {
                max = z_out[i];
                best_z = i;
            }
        }

        std::cout << "Predicted: " << best_z << ", Actual: " << test_label[inputIndex] << std::endl;

    }

    return 0;
}

void init()
{
    std::string fname;
    std::cout << "Enter filename for model file (e.g. path/to/epoch#_correct####.txt): ";
    std::cin >> fname;
    std::ifstream model;
    model.open(fname.c_str());
    if (model.fail())
    {
        std::cout << "Failed to open test file\n";
        return;
    }

    for (int i = 0; i<6; i++)
    {
        for (int j = 0; j<5; j++)
        {
            for (int k = 0; k<5; k++)
            {
                model >> w_ic[i][j][k];
            }
        }
    }
    for (int i = 0; i<6; i++)
    {
        for (int j = 0; j<24; j++)
        {
            for (int k = 0; k<24; k++)
            {
                model >> b_c[i][j][k];
            }
        }
    }
    for (int i = 0; i<6; i++)
    {
        for (int j = 0; j<12; j++)
        {
            for (int k = 0; k<12; k++)
            {
                model >> w_cm[i][j][k];
            }
        }
    }
    for (int i = 0; i<6; i++)
    {
        for (int j = 0; j<12; j++)
        {
            for (int k = 0; k<12; k++)
            {
                for (int l = 0; l < 45; l++)
                {
                    model >> u[i][j][k][l];
                } 
            }
        }
    }
    for (int i = 0; i < 45; i++)
    {
        model >> b_sig[i];
        for (int j = 0; j < 10; j++)
            model >> v[i][j];
    }
    for (int i = 0; i < 10; i++)
                model >> b_out[i];
    model.close();
}

void loadData()
{
    unsigned char byte;

    std::cout << "Loading testing data...";
    std::ifstream testingFile("t10k-images-idx3-ubyte", std::ios::in | std::ios::binary);
    testingFile.seekg(16);
    std::ifstream testLabelFile("t10k-labels-idx1-ubyte", std::ios::in | std::ios::binary);
    testLabelFile.seekg(8);

    for (int n=0; n<10000; n++){
        for (int i=0; i<28; i++)
        {
            for (int j=0; j<28; j++){
                testingFile.read((char*)&byte, 1);
                test[n][i][j] = (float)(int)byte/255.0f;
            }
        }
        testLabelFile.read((char*)&byte, 1);
        test_label[n] = (int)byte;
    }

    std::cout << " done!\n";
}

float max(float a, float b, float c, float d)
{
    float left = fmax(a,b);
    float right = fmax(c,d);
    return fmax(left, right);
}

float sigmoid(float s)
{
    return 1.0f/(1.0f + exp(-s));
}

float deltaSigmoid(float z)
{
    return z*(1.0f-z);
}

/*
float input[28][28] = {0};    //input
float w_ic[6][5][5] = {0};    //input->convolution weights
float b_c[6][24][24] = {0};   //convolution bias
float y_c[6][24][24] = {0};   //convolution layer activation output
float w_cm[6][12][12] = {0};  //conv->max layer weights
float y_m[6][12][12] = {0};   //max layer activation output
float u[6][12][12][45] = {0}; //max->sigmoid layer weights
float b_sig[45] = {0};        //sigmoid layer bias
float y_sig[45] = {0};        //sigmoid layer activation output
float v[45][10] = {0};        //sigmoid->output layer weights
float b_out[10] = {0};        //output layer bias
float z_out[10] = {0};        //output
float t[10] = {0};            //truth values
float E[10] = {0};            //error per output node
*/
void evaluate()
{
    //SIGMOID CONVOLUTION LAYER
    for (int page = 0; page < 6; page++)
    {
        for (int pagex = 0; pagex < 24; pagex++)
        {
            for (int pagey = 0; pagey < 24; pagey++)
            {
                float sum = b_c[page][pagex][pagey];
                for (int ix=0; ix < 5; ix++){
                    for (int iy=0; iy < 5; iy++)
                        sum += input[ix+pagex][iy+pagey] * w_ic[page][ix][iy];
                }
                y_c[page][pagex][pagey] = sigmoid(sum);
            }
        }
    }

    //MAX LAYER
    for (int page = 0; page < 6; page++)
        for (int pagex = 0; pagex < 12; pagex++)
            for (int pagey = 0; pagey < 12; pagey++)
                y_m[page][pagex][pagey] = max(
                        y_c[page][pagex*2][pagey*2],
                        y_c[page][pagex*2+1][pagey*2],
                        y_c[page][pagex*2][pagey*2+1],
                        y_c[page][pagex*2+1][pagey*2+1]
                    ) * w_cm[page][pagex][pagey];


    //SIGMOID LAYER
    for (int node = 0; node < 45; node++)
    {
        float sum = b_sig[node];
        for (int page = 0; page < 6; page++)
            for (int px = 0; px < 12; px++)
                for (int py = 0; py < 12; py++)
                    sum += y_m[page][px][py] * u[page][px][py][node];

        y_sig[node] = sigmoid(sum);
    }

    //OUTPUT LAYER
    for (int node = 0; node < 10; node++)
    {
        float sum = b_out[node];
        for (int in = 0; in < 45; in++)
            sum += y_sig[in] * v[in][node];

        z_out[node] = sigmoid(sum);
        E[node] = 0.5 * (z_out[node] - t[node]) * (z_out[node] - t[node]);
    }


}

void descend(float eta)
{
    //OUTPUT LAYER
    for (int node = 0; node < 10; node++)
    {
        float gradient0 = eta*(z_out[node] - t[node])*deltaSigmoid(z_out[node]);
        b_out[node] = b_out[node] - gradient0;

        //SIGMOID LAYER
        for (int signode = 0; signode < 45; signode++)
        {
            float gradient1 = gradient0 * v[signode][node] * deltaSigmoid(y_sig[signode]);
            b_sig[signode] = b_sig[signode] - gradient1;
            v[signode][node] = v[signode][node] - gradient0 * y_sig[signode];
            
            //MAX LAYER
            
            for (int maxnode = 0; maxnode < 6; maxnode++)
            {
                for (int mx = 0; mx < 12; mx++)
                {
                    for (int my = 0; my < 12; my++)
                    {
                        float gradient2 = gradient1 * u[maxnode][mx][my][signode];

                        //float relevant_y_c = y_m[maxnode][mx][my] / w_cm[maxnode][mx][my];
                        float relevant_y_c = max(
                            y_c[maxnode][mx*2][my*2],
                            y_c[maxnode][mx*2+1][my*2],
                            y_c[maxnode][mx*2][my*2+1],
                            y_c[maxnode][mx*2+1][my*2+1]
                        );
                        w_cm[maxnode][mx][my] = w_cm[maxnode][mx][my] - gradient2 * relevant_y_c;

                        float gradient3 = gradient2 * w_cm[maxnode][mx][my] * deltaSigmoid(relevant_y_c);
                        b_c[maxnode][mx][my] = b_c[maxnode][mx][my] - gradient3;
                        u[maxnode][mx][my][signode] = u[maxnode][mx][my][signode] - gradient1 * y_m[maxnode][mx][my];
                        
                        for (int cx_ = 0; cx_ < 2; cx_++)
                        {
                            for (int cy_ = 0; cy_ < 2; cy_++)
                            {
                                int cx = mx*2 + cx_;
                                int cy = my*2 + cy_;
                                //Backpropagate only through the maximum of the 4 convolution nodes
                                if (y_c[maxnode][cx][cy] == relevant_y_c)
                                {
                                    for (int ix=0; ix < 5; ix++)
                                    {
                                        for (int iy=0; iy < 5; iy++)
                                            w_ic[maxnode][ix][iy] = w_ic[maxnode][ix][iy] - gradient3 * input[ix+cx][iy+cy];
                                    }
                                }
                            }
                        }
                        

                    }
                }
            }
            
        }
    }
}

void outputWeights(int epoch, int testCorrect)
{
    std::ostringstream oss;
    oss << "epoch" << epoch << "_correct" << testCorrect << ".txt";
    std::cout << "Outputting weight file: " << oss.str() << std::endl;
    
    std::ofstream weightfile (oss.str().c_str());

    if (weightfile.is_open())
    {
        for (int i = 0; i<6; i++)
        {
            for (int j = 0; j<5; j++)
            {
                for (int k = 0; k<5; k++)
                {
                    weightfile << w_ic[i][j][k] << std::endl;
                }
            }
        }
        for (int i = 0; i<6; i++)
        {
            for (int j = 0; j<24; j++)
            {
                for (int k = 0; k<24; k++)
                {
                    weightfile << b_c[i][j][k] << std::endl;
                }
            }
        }
        for (int i = 0; i<6; i++)
        {
            for (int j = 0; j<12; j++)
            {
                for (int k = 0; k<12; k++)
                {
                    weightfile << w_cm[i][j][k] << std::endl;
                }
            }
        }
        for (int i = 0; i<6; i++)
        {
            for (int j = 0; j<12; j++)
            {
                for (int k = 0; k<12; k++)
                {
                    for (int l = 0; l < 45; l++)
                    {
                        weightfile << u[i][j][k][l] << std::endl;
                    } 
                }
            }
        }
        for (int i = 0; i < 45; i++)
        {
            weightfile << b_sig[i] << std::endl;
            for (int j = 0; j < 10; j++)
                weightfile << v[i][j] << std::endl;
        }
        for (int i = 0; i < 10; i++)
                    weightfile << b_out[i] << std::endl;
    

        weightfile.close();
    }
}

int checkTestAccuracy()
{
    int numCorrect = 0;
    for (int n = 0; n < 10000; n++)
    {
        //Set inputs
        for (int x = 0; x < 28; x++)
            for (int y = 0; y < 28; y++)
                input[x][y] = test[n][x][y];

        //Set truth values
        for (int i = 0; i < 10; i++)
        {
            t[i] = 0;
            if (i == test_label[n])
                t[i] = 1;
        }

        //Feed-forward
        evaluate();

        //Check if correct
        float max = 0;
        int best_z = 0;
        for (int i = 0; i < 10; i++)
        {
            if (z_out[i] > max)
            {
                max = z_out[i];
                best_z = i;
            }
        }

        if (best_z == test_label[n])
        {
            numCorrect++;
        }
    }
    return numCorrect;
}
