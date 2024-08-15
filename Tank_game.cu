#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here


__device__ int round_count;

__global__ void health_initialization(int* health_device, int health)
{

    health_device[threadIdx.x] = health;
    if (threadIdx.x == 0)
    {
        round_count = 1;
    }
}

__global__ void closest_Tank_hit(int* x_device, int* y_device, int* health_device, int T, unsigned long long int* hitList)
{



    long long int Y_axis = y_device[(blockIdx.x + round_count) % T] - y_device[blockIdx.x];
    long long int X_axis = x_device[(blockIdx.x + round_count) % T] - x_device[blockIdx.x];


    if (threadIdx.x == 0)
    {
        hitList[blockIdx.x] = LLONG_MAX;
    }
    __syncthreads();

    if (threadIdx.x == blockIdx.x || health_device[threadIdx.x] <= 0 || health_device[blockIdx.x] <= 0)
    {
        return;
    }

    long long int Initial_X = x_device[threadIdx.x] - x_device[blockIdx.x];
    long long int Initial_Y = y_device[threadIdx.x] - y_device[blockIdx.x];
    int tank_count = 0;

    if (X_axis * Initial_Y == Y_axis * Initial_X && X_axis * Initial_X >= 0 && Y_axis * Initial_Y >= 0)
    {
        unsigned long long temp_dist = ((unsigned long long int)(abs(Initial_Y)) + ((unsigned long long int)(abs(Initial_X))) << 32);
        tank_count++;
        unsigned long long temp_valX = temp_dist | ((unsigned long long)threadIdx.x);
        atomicMin(&hitList[blockIdx.x], temp_valX);
    }
}

__global__ void score_updating(int* device_health, int* scoreList, int* over, int T, unsigned long long* target_arr)
{

    if (threadIdx.x == 0)
    {
        round_count++;
        if (round_count % T == 0)
        {
            round_count = 1;
        }
    }
    int a = 1;
    if (target_arr[threadIdx.x] != (LLONG_MAX)) {
        int tar_tank = int(target_arr[threadIdx.x] & (LLONG_MAX));

        int previous_hth = atomicSub(&device_health[tar_tank], a);

        if (previous_hth == a)
        {
            atomicAdd(over, a);
        }
        scoreList[threadIdx.x] = scoreList[threadIdx.x] + a;
    }
}

//***********************************************


int main(int argc, char** argv)
{
    // Variable declarations
    int M, N, T, H, * xcoord, * ycoord, * score;


    FILE* inputfilepointer;

    //File Opening for read
    char* inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL) {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int*)malloc(T * sizeof(int));  // X coordinate of each tank
    ycoord = (int*)malloc(T * sizeof(int));  // Y coordinate of each tank
    score = (int*)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++)
    {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int* Score_device, * health_device;
    cudaMalloc(&health_device, T * sizeof(int));
    cudaMalloc(&Score_device, T * sizeof(int));
    cudaMemset(health_device, 0, T * sizeof(int));
    cudaMemset(Score_device, 0, T * sizeof(int));




    health_initialization << <1, T >> > (health_device, H);
    cudaDeviceSynchronize();

    unsigned long long* device_hitArr;
    cudaMalloc(&device_hitArr, T * sizeof(unsigned long long int));
    int host_over_value = 0;

    int* x_device, * device_over_value, * y_device;
    cudaMalloc(&device_over_value, 1 * sizeof(int));
    cudaMalloc(&x_device, T * sizeof(int));
    cudaMalloc(&y_device, T * sizeof(int));
    cudaMemcpy(x_device, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(device_over_value, 0, 1 * sizeof(int));
    cudaMemcpy(y_device, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    int blocksize = T, threadsperblock = T;
    int check = 0;
    while (1)
    {
        if (host_over_value < T - 1) {
            closest_Tank_hit << <blocksize, threadsperblock >> > (x_device, y_device, health_device, T, device_hitArr);
            cudaDeviceSynchronize();
            check++;
            score_updating << <1, threadsperblock >> > (health_device, Score_device, device_over_value, T, device_hitArr);
            cudaDeviceSynchronize();
            check--;
            cudaMemcpy(&host_over_value, device_over_value, sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
        else
            break;
    }

    cudaMemcpy(score, Score_device, T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(y_device);
    cudaFree(x_device);
    cudaFree(health_device);
    
    

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char* outputfilename = argv[2];
    char* exectimefilename = argv[3];
    FILE* outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++)
    {
        fprintf(outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}
