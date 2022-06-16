
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

#define ANIMATION true
#define ANIMATIONFILE "Brennkammer.gif"
#define WRITESURFACE true
#define SURFACEFILE "Surface.txt"

#define CHAMBERWIDTH 10
#define CHAMBERHEIGHT 10
#define SIMSTEPS 25

cudaError_t simStep(int* chamber, int* newchamber, int* burncount);

void printChamber();

void makeImage(std::string& filename);

const int chamberArrLen = (CHAMBERHEIGHT + 2) * (CHAMBERWIDTH + 2);
int chamber[chamberArrLen] = { 0 };
int burnSurfaces[SIMSTEPS];

__global__ void countKernel(int* chamber, int* countArr) {
    int count = 0;
    int rowIndex = (threadIdx.x + 1) * (CHAMBERWIDTH + 2);
    for (int i = 1; i < CHAMBERWIDTH+1; i++) {
        if (chamber[rowIndex + i] == BURN2) {
            countArr[threadIdx.x]++;
        }
    }
}

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
        if (chamber[cellnumber + CHAMBERWIDTH + 2] == BURN1 
            || chamber[cellnumber + 1] == BURN1 
            || chamber[cellnumber - CHAMBERWIDTH - 2] == BURN1
            || chamber[cellnumber - 1] == BURN1) {
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
    std::fill_n(chamber, chamberArrLen, NEW);
    chamber[CHAMBERWIDTH * 2] = BURN1;

    cudaError_t cudaStatus;

    char fileno[5];
    std::string filename;

    int burncount = 0;

    int step = 0;
    while (step < SIMSTEPS) {
        burncount = 0;
        cudaStatus = simStep(chamber, 0, &burncount);
        burnSurfaces[step] = burncount;

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "simulation failed at Step %d", step);
            return 1;
        }

        printf("%d : %d Elements burning\n", step, burncount);

        if (!ANIMATION) {
            step++;
            continue;
        }
        
        snprintf(fileno, 5, "%04d", step);
        filename = "Brennkammer-";
        filename.append(fileno);

        std::cout << step << filename << "\n";
        makeImage(filename);
        step++;
    }

    std::cout << std::endl;
    if (ANIMATION) {
        system("move *.png Simulation");
        //system("magick convert Simulation\\*.ppm Simulation\\.png");
        system("del *.ppm");

        std::cout << "\nFiles have been converted to png \n";

        system("magick convert -delay 5 Simulation\\Brennkammer-*.png Simulation\\Brennkammer.gif");
        system("del Simulation\\*.png");

        std::cout << "Animation was created\n";
    }
    if (WRITESURFACE) {
        std::ofstream surfacefile("Simulation\\" + std::string(SURFACEFILE));
        for (int i = 0; i < SIMSTEPS; i++)
        {
            surfacefile << burnSurfaces[i] << ";";
        }
        surfacefile.close();
        system("python PlotSurfaceArea.py");
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }
    return 0;
}

void makeImage(std::string& filename)
{

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
    system(("magick convert " + filename + ".ppm " + filename + ".png").c_str());
}

void printChamber()
{
    for (int y = 0; y < CHAMBERHEIGHT; y++) {
        for (int x = 0; x < CHAMBERWIDTH; x++)
        {
            std::cout << " " << chamber[((y + 1) * (CHAMBERWIDTH + 2)) + x + 1];
        }
        std::cout << std::endl;
    }
}

cudaError_t simStep(int* chamber, int* newchamber, int* burncount) {
    int* dev_chamber = 0;
    int* dev_newchamber = 0;
    int* dev_burncounts = 0;
    int burncounts[CHAMBERHEIGHT];
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0); //set GPU
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        return cudaStatus;
    }

#pragma region Memory Allocation
    cudaStatus = cudaMalloc((void**)&dev_chamber, chamberArrLen * sizeof(int)); //Allocate Memory for current chamber
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memory Allocation of CHAMBER Failed\n");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        cudaFree(dev_burncounts);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_newchamber, chamberArrLen * sizeof(int)); //Allocate Memory for next Chamber
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memory Allocation of NEWCHAMBER Failed\n");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        cudaFree(dev_burncounts);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_burncounts, CHAMBERWIDTH * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memory Allocation of BURNCOUNTS failed\n");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        cudaFree(dev_burncounts);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_chamber, chamber, chamberArrLen * sizeof(int), cudaMemcpyHostToDevice); //copy current chamber into Device Memory
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memory Transfer from Host to Device Failed\n");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        cudaFree(dev_burncounts);
        return cudaStatus;
    }

#pragma endregion

    simKernel <<<CHAMBERHEIGHT, CHAMBERWIDTH >>> (dev_chamber, dev_newchamber);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Simulation Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        cudaFree(dev_burncounts);
        return cudaStatus;
    }

    countKernel << < 1, CHAMBERHEIGHT >> > (dev_newchamber, dev_burncounts);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Counting Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        cudaFree(dev_burncounts);
        return cudaStatus;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        cudaFree(dev_burncounts);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(chamber, dev_newchamber, chamberArrLen * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memory Transfer of new Chamber from Device to hostFailed\n");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        cudaFree(dev_burncounts);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(burncounts, dev_burncounts, CHAMBERHEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Memory Transfer of burncounts from Device to hostFailed\n");
        cudaFree(dev_chamber);
        cudaFree(dev_newchamber);
        cudaFree(dev_burncounts);
        return cudaStatus;
    }

    for (int i = 0; i < CHAMBERHEIGHT; i++) {
        burncount[0] += burncounts[i];
    }

    return cudaStatus;

}
