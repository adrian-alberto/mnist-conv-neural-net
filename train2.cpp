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
float w_mh[8][12][12][45] = {0};
float b_h[45] = {0};
float y_h[45] = {0};
float w_ho[45][10] = {0};
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
float activate(float);
float dActivate(float);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

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
                numCorrect++;

            //Back-propagate
            descend(learning_rate);
		}
        std::cout << "epoch " << epoch << ": " << ((float) numCorrect)/600.0f << "% training accuracy" << std::endl;
	}


	return 0;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

void initRandom()
{
    srand(time(0));
    for (int i = 0; i<8; i++)
    {
        for (int j = 0; j<5; j++)
        {
            for (int k = 0; k<5; k++)
            {
                w_ic[i][j][k] = (float) rand() / (float)(RAND_MAX) - 0.5;
            }
        }
    }
    for (int i = 0; i<8; i++)
    {
        b_c[i] = (float) rand() / (float)(RAND_MAX) - 0.5;
        for (int j = 0; j < 12; j++)
            for (int k = 0; k < 12; k++)
                w_mh[i][j][k];
    }

    for (int i = 0; i < 45; i++)
    {
        b_h[i] = (float) rand() / (float)(RAND_MAX) - 0.5;
        for (int j = 0; j < 10; j++)
            w_ho[i][j] = (float) rand() / (float)(RAND_MAX) - 0.5;
    }
    for (int i = 0; i < 10; i++)
        b_o[i] = (float) rand() / (float)(RAND_MAX) - 0.5;
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
                train_input[n][i][j] = (float)(int)byte/255.0f;
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
                test_input[n][i][j] = (float)(int)byte/255.0f;
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
                        s += (input[cx+ix][cy+iy] * w_ic[frame][ix][iy]);
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
                        y_c[frame][mx*2][my*2],
                        y_c[frame][mx*2][my*2+1],
                        y_c[frame][mx*2+1][my*2],
                        y_c[frame][mx*2+1][my*2+1]
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
    float G_h[45] = {0};
    for (int j = 0; j < 10; j++)
    {
        float g0 = (z_o[j] - t[j]) * dActivate(z_o[j]);
        b_o[j] -= eta*g0;
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
        b_h[i] -= eta*G_h[i];
    }

    for (int frame = 0; frame < 8; frame++)
    {
        float G_c = 0; //total gradient at the conv layer

        for (int mx = 0; mx < 12; mx++)
        {
            for (int my = 0; my < 12; my++)
            {
                float G_m = 0;
                for (int i = 0; i < 45; i++)
                {
                    G_m += w_mh[frame][mx][my][i] * G_h[i];
                    w_mh[frame][mx][my][i] -= eta*G_h[i]*y_m[frame][mx][my];
                }

                for (int cx = mx; cx < mx+2; cx++)
                {
                    for (int cy = my; cy < my+2; cy++)
                    {
                        if (y_c[frame][cx][cy] == y_m[frame][mx][my])
                        {
                            //g_c is the gradient at this convolutional layer at (cx,cy)
                            //float g_c = G_m;
                            //G_c += g_c;

                            /*for (int ix = cx; ix < cx+5; ix++)
                                for (int iy = cy; iy < cy+5; iy++)
                                    w_ic[frame][ix-cx][iy-cy] -= eta*input[ix][iy]*g_c * dActivate(y_c[frame][cx][cy]);*/

                        }
                    }
                }
            }
        }

        b_c[frame] -= eta*G_c;



    }





}



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

float Relu(float s)
{
    if (s > 0)
        return s;
    else
        return 0.01*s;
}

float deltaRelu(float z)
{
    if (z > 0)
        return 1;
    else
        return 0.01;
}

//sigmoid
float activate(float s)
{
    return 1.0f/(1.0f + exp(-s));
}
//delta sigmoid
float dActivate(float z)
{
    return z*(1.0f-z);
}

float max(float a, float b, float c, float d)
{
    float left = fmax(a,b);
    float right = fmax(c,d);
    return fmax(left, right);
}