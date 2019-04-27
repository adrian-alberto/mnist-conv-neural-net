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


int max_epoch = 16;
float learning_rate = 0.2;


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

float input[28][28] = {0};
float w_ic[8][5][5] = {0};
float b_c[8] = {0};
float y_c[8][24][24] = {0};
float y_m[8][12][12] = {0};
float w_ms[8][12][12][45] = {0};
float b_s[45] = {0};
float y_s[45] = {0};
float w_so[45][10] = {0};
float b_o[10] = {0};
float z_o[10] = {0};
float t[10] = {0};

float train_input[60000][28][28] = {0};
float train_label[60000] = {0};
float test_input[10000][28][28] = {0};
float test_label[10000] = {0};

void loadWeights();
void initRandom();
int loadData();
int evaluate(); //returns 1 if correct, 0 if incorrect
void descend(float);
float max(float,float,float,float);
void outputWeights(int,int);

int main()
{
	if (loadData() == 1)
		return 0;

	initRandom();

	for (int epoch = 0; epoch < max_epoch; epoch++)
	{
		int numCorrect = 0;
		for (int n = 0; n < 60000; n++)
		{
            if (n % 1000 == 0)
                std::cout << "Progress: " << n << "/60000 ... " <<std::endl;

            //Set inputs
            for (int x = 0; x < 28; x++)
                for (int y = 0; y < 28; y++)
                    input[x][y] = train[n][x][y];

            //Set truth values
            for (int i = 0; i < 10; i++)
            {
                t[i] = 0;
                if (i == train_label[n])
                    t[i] = 1;
            }

			numCorrect += evaluate();
		}
	}


	return 0;
}

void initRandom()
{
    srand(time(0));
    for (int i = 0; i<6; i++)
    {
        for (int j = 0; j<5; j++)
        {
            for (int k = 0; k<5; k++)
            {
                w_ic[i][j][k] = (float) rand() / (float)(RAND_MAX) - 0.5;
            }
        }
    }
    for (int i = 0; i<6; i++)
    {
        for (int j = 0; j<24; j++)
        {
            for (int k = 0; k<24; k++)
            {
                b_c[i][j][k] = (float) rand() / (float)(RAND_MAX) - 0.5;
            }
        }
    }
    for (int i = 0; i<6; i++)
    {
        for (int j = 0; j<12; j++)
        {
            for (int k = 0; k<12; k++)
            {
                w_cm[i][j][k] = (float) rand() / (float)(RAND_MAX) - 0.5;
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
                    u[i][j][k][l] = (float) rand() / (float)(RAND_MAX) - 0.5;
                } 
            }
        }
    }
    for (int i = 0; i < 45; i++)
    {
        b_sig[i] = (float) rand() / (float)(RAND_MAX) - 0.5;
        for (int j = 0; j < 10; j++)
            v[i][j] = (float) rand() / (float)(RAND_MAX) - 0.5;
    }
    for (int i = 0; i < 10; i++)
                b_out[i] = (float) rand() / (float)(RAND_MAX) - 0.5;
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

    for (int n=0; n<60000; n++){
        for (int i=0; i<28; i++)
        {
            for (int j=0; j<28; j++){
                trainingFile.read((char*)&byte, 1);
                train[n][i][j] = (float)(int)byte/255.0f;
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
    return 0;
}




int evaluate()
{
	
}