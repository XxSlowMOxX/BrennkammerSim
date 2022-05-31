
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <algorithm>
#include <string>

#define NOTHING 0
#define NEW 1
#define BURN1 2
#define BURN2 3
#define BURNT 4

#define CHAMBERWIDTH 100
#define CHAMBERHEIGHT 100
#define SIMSTEPS 500

cudaError_t simStep(int* chamber, int* newchamber);

const int chamberArrLen = (CHAMBERHEIGHT + 2) * (CHAMBERWIDTH + 2);

__global__ void simKernel(int* chamber, int* newchamber) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int cellnumber = ((y+1) * (CHAMBERWIDTH+2)) + x + 1;
    switch (chamber[cellnumber]) {
    case NOTHING:
        newchamber[cellnumber] = NOTHING;
        return;
        break;
    case NEW:
        newchamber[cellnumber] = NEW;
        if (chamber[cellnumber + CHAMBERWIDTH + 2] == BURN2 
            || chamber[cellnumber + 1] == BURN2 
            || chamber[cellnumber - CHAMBERWIDTH - 2] == BURN2
            || chamber[cellnumber - 1] == BURN2) {
            newchamber[cellnumber] = BURN1;
        }
        return;
        break;
    case BURN1:
        newchamber[cellnumber] = BURN2;
        return;
        break;
    default: //Wenn BURN2 oder BURNT
        newchamber[cellnumber] = BURNT;
        return;
        break;
    }
}

int main()
{
    
    int chamber[chamberArrLen] = { 0 };
    std::fill_n(chamber, chamberArrLen, NEW);
    chamber[CHAMBERWIDTH * 2] = BURN1;

    /*
    for (int y = 0; y < CHAMBERHEIGHT; y++) {
        for (int x = 0; x < CHAMBERWIDTH; x++)
        {
            std::cout << " " << chamber[((y + 1) * (CHAMBERWIDTH+2)) + x + 1];
        }
        std::cout << std::endl;
    }
    */

    cudaError_t cudaStatus;

    std::string filename;

    int step = 0;
    while (step < SIMSTEPS) {
        cudaStatus = simStep(chamber, 0);

        filename = "Brennkammer-" + std::to_string(step);

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "simulation failed at Step %d", step);
            return 1;
        }

        std::ofstream img(filename + ".ppm");
        img << "P3\n";
        img << std::to_string(CHAMBERHEIGHT) << " " << std::to_string(CHAMBERWIDTH) << std::endl;
        img << "255" << std::endl;

        for (int y = 0; y < CHAMBERHEIGHT; y++) {
            for (int x = 0; x < CHAMBERWIDTH; x++)
            {
                switch (chamber[((y + 1) * (CHAMBERWIDTH + 2)) + x + 1]) {
                case NOTHING:
                    img << "255 255 255 \n";
                    break;
                case NEW:
                    img << "48 0 0\n";
                    break;
                case BURN1:
                    img << "166 111 2\n";
                    break;
                case BURN2:
                    img << "255 106 0\n";
                    break;
                case BURNT:
                    img << "0 0 0\n";
                    break;
                default:
                    img << "14 237 14\n";
                    break;
                }
            }
        }
        img.close();
        step++;
    }

    std::cout << std::endl;

    system("move Brennkammer-*.ppm Simulation");
    system("magick convert Simulation\\Brennkammer-*.ppm Simulation\\Brennkammer.png");
    system("del Simulation\\Brennkammer-*.ppm");

    std::cout << "\nFiles have been converted to png \n";

    system("magick convert -delay 25 Simulation\\Brennkammer-*.png Simulation\\Brennkammer.gif");
    system("del Simulation\\Brennkammer-*.png");

    std::cout << "Animation was created\n";

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}

cudaError_t simStep(int* chamber, int* newchamber) {
    int* dev_chamber = 0;
    int* dev_newchamber = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0); //set GPU
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        return cudaStatus;
    }

#pragma region Memory Allocation
    cudaStatus = cudaMalloc((void**)&dev_chamber, chamberArrLen * sizeof(int)); //Allocate Memory for current chamber
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memory Allocation Failed");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_newchamber, chamberArrLen * sizeof(int)); //Allocate Memory for next Chamber
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memory Allocation Failed");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_chamber, chamber, chamberArrLen * sizeof(int), cudaMemcpyHostToDevice); //copy current chamber into Device Memory
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memory Transfer from Host to Device Failed");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        return cudaStatus;
    }

#pragma endregion

    simKernel <<<CHAMBERHEIGHT, CHAMBERWIDTH >>> (dev_chamber, dev_newchamber);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        return cudaStatus;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(chamber, dev_newchamber, chamberArrLen * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memory Transfer from Device to hostFailed");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        return cudaStatus;
    }

    return cudaStatus;

}
