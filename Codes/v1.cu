#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#define PI 3.1415926536
#define e 2.718281828459

#define N 64*64
#define THREADS_PER_BLOCK 64

struct timeval tic(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv;
}

typedef struct Patches
{
    int index;
    float central;
    float* patchArray;
}Patch;

double toc(struct timeval begin){
    struct timeval end;
    gettimeofday(&end,NULL);
    double stime = ((double)(end.tv_sec-begin.tv_sec)*1000)+((double)(end.tv_usec-begin.tv_usec)/1000);
    stime = stime / 1000;
    return (stime);
}

float* readFile(int n, int m, char *file_path){
    FILE* ptrFile = fopen(file_path, "r");

    float *I = (float*)malloc(n*m*sizeof(float));

    if (!ptrFile){
        printf("Error Reading File\n");
        exit (0);
    }

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            fscanf(ptrFile,"%f,", &I[n*i+j]);
        }
    }

    fclose(ptrFile);

    return I;
}

void toTXT(float* array,char *output, int n, int m){
    FILE *fp;

    fp=fopen(output,"w");

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            if(j<m-1){
                fprintf(fp,"%lf,",array[n*i+j]);
            }else if(j==m-1){
                fprintf(fp,"%lf",array[n*i+j]);
            }
        }
        fprintf(fp,"\n",array[n*i]);
    }
    fclose(fp);
    printf("File %s saved.\n", output);
}

__global__ void normalization(float* A, float* B, float min, float max){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)
        B[i] = (A[i] - min) / max;
}

float AWGN_generator()     //https://www.embeddedrelated.com/showcode/311.php
{/* Generates additive white Gaussian Noise samples with zero mean and a standard deviation of 1. */
  float dev = 0.03162; //var = 0.01
  float temp1;
  float temp2;
  float result;
  int p = 1;

  while( p > 0 )
  {
	temp2 = ( rand() / ( (float)RAND_MAX ) ); /*  rand() function generates an
                                                       integer between 0 and  RAND_MAX,
                                                       which is defined in stdlib.h.
                                                   */

    if ( temp2 == 0 )
    {// temp2 is >= (RAND_MAX / 2)
        p = 1;
    }// end if
    else
    {// temp2 is < (RAND_MAX / 2)
        p = -1;
    }// end else

  }// end while()

  temp1 = cos( ( 2.0 * (float)PI ) * rand() / ( (float)RAND_MAX ) );
  result = sqrt( -2.0 * log( temp2 ) ) * temp1;

  return result * dev;	// return the generated random sample to the caller

}// end AWGN_generator()

Patch* makePatches(float* J, int n, int m, Patch* allPatches, int patchSizeH, int patchSizeW){
    int mdW = (patchSizeW - 1)/2;
    int mdH = (patchSizeH - 1)/2;
    
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){

            for(int w=0; w<patchSizeW; w++){
                for(int h=0; h<patchSizeH; h++){
                    allPatches[n*j+i].patchArray[patchSizeH*w+h] = 0;
                }
            }

            allPatches[n*j+i].central = J[n*j+i];
            allPatches[n*j+i].index = n*j+i;

            if(i==0 && j==0){
                for(int w=mdW; w<patchSizeW; w++){
                    for(int h=mdH; h<patchSizeH; h++){
                        allPatches[n*j+i].patchArray[patchSizeH*w+h] = J[(n*j+i)-(mdW-w)*n-(mdH-h)];
                    }
                } 
            }else if(i>0 && j==0){
                for(int h=0; h<patchSizeH-1; h++){
                    for(int w=0; w<patchSizeW; w++){
                        allPatches[n*j+i].patchArray[patchSizeH*w+h] = allPatches[n*j+(i-1)].patchArray[patchSizeH*w+(h+1)];
                    }
                }
               

                for(int w=mdW; w<patchSizeW; w++){
                    if((n-1-i) >= mdH){
                        allPatches[n*j+i].patchArray[patchSizeH*w+(patchSizeH-1)] = J[(n*j+i)-(mdW-w)*n+mdH];
                    }else if((n-1-i) < mdH){
                        allPatches[n*j+i].patchArray[patchSizeH*w+(patchSizeH-1)] = 0;
                    }
                }
            }else if(j>0){
                for(int w=0; w<patchSizeW-1; w++){
                    for(int h=0; h<patchSizeH; h++){
                        allPatches[n*j+i].patchArray[patchSizeH*w+h] = allPatches[n*(j-1)+i].patchArray[patchSizeH*(w+1)+h];
                    }
                }

                int a,b;
                if(i>=mdH && (n-1-i)>=mdH){
                    a = 0;
                    b = patchSizeH;
                }else if(i<mdH && (n-1-i)>=mdH){
                    a = mdH - i;
                    b = patchSizeH;
                }else if(i<mdH && (n-1-i)<mdH){
                    a = mdH - i;
                    b = mdH + (n-i);
                }else if(i>=mdH && (n-1-i)<mdH){
                    a = 0;
                    b = mdH + (n-i);
                }

                for(int h=a; h<b; h++){
                    if((m-1-j) >= mdW){
                        allPatches[n*j+i].patchArray[patchSizeH*(patchSizeW-1)+h] = J[(n*j+i)+mdW*n-(mdH-h)];
                    }else if((m-1-j) < mdW){
                        allPatches[n*j+i].patchArray[patchSizeH*(patchSizeW-1)+h] = 0;
                    }
                }
            }
        }
    }
    return allPatches;
}

float* computeG_a(int patchSizeH, int patchSizeW, float patchSigma){
    float* gauss = (float*)malloc(patchSizeH*patchSizeW*sizeof(float));
    
    for (int i = 0; i < patchSizeH; i++) {
        for (int j = 0; j < patchSizeW; j++) {
            float y = i - (patchSizeH - 1) / 2.0;
            float x = j - (patchSizeW - 1) / 2.0;
            gauss[patchSizeW*i+j] = (1/2.0) * exp(-(x * x + y * y) / (2.0 * PI * patchSigma * patchSigma));
        }
    }    
    return gauss;
}

__global__ void dist(float *W,float *p_i, float *P, float *V, int pSize, float filtSigma){
    float d = 0;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(k<N){
        for (int i = 0; i < pSize; i++) {
            d += V[i] * pow(p_i[i] - P[pSize*k+i],2);
        }
        d = sqrt(d);
        W[k] = exp(-pow(d,2) / filtSigma);
        d=0;
    }   
}

__global__ void dim(float *w, float *z){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<N){
        w[i] = w[i] / *z;
    }
}

int main(int argc, char *argv[]){
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int patchSizeH = atoi(argv[3]);
    int patchSizeW = atoi(argv[4]);

    char* file_path;
    file_path=(char*)malloc(strlen(argv[5])*sizeof(char));
    memcpy(file_path,argv[5],strlen(argv[5]));
       
    float patchSigma =5/3;
    float filtSigma =0.01 ;
    
    float *I, *I_norm, *J, *If;
    float *dev_I, *dev_I_norm, *dev_J, *dev_gauss;
    float *P, *dev_P, *w;

    int size = N * sizeof(float);
    int sizePatch = patchSizeH * patchSizeW * sizeof(float);
    int pSize = patchSizeH * patchSizeW;
    
    //allocate memory for device copies
    cudaMalloc(&dev_I, size);
    cudaMalloc(&dev_I_norm, size);
    cudaMalloc(&dev_J, size);
    cudaMalloc(&dev_gauss, sizePatch);
    cudaMalloc(&dev_P, N*pSize*sizeof(float));    

    I = (float*)malloc(size);
    I_norm = (float*)malloc(size);
    J = (float*)malloc(size);
    If = (float*)malloc(size);
    
    Patch* allPatches;
    allPatches = (Patch*)malloc(n*m*sizeof(Patch));

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            allPatches[n*j+i].patchArray = (float*)malloc(patchSizeH*patchSizeW*sizeof(float));
        }
    }    

    w = (float*)malloc(N*N*sizeof(float));
    float* gauss = (float*)malloc(sizePatch);
    float* Z = (float*)malloc(size);

    struct timeval tStart;

    I = readFile(n,m,file_path);

    //find min of 'I' and max of 'I-min'
    float min = INFINITY;
    float max = -1.0;

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            if(I[n*i+j]<min) min= I[n*i+j];
        }
    }    

    for(int i=0; i<n*m; i++){
        if((I[i]-min)>max) max = I[i]-min;
    }

    cudaMemcpy(dev_I, I, size, cudaMemcpyHostToDevice);
    normalization<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dev_I, dev_I_norm, min, max);
    cudaMemcpy(I_norm, dev_I_norm, size, cudaMemcpyDeviceToHost);
   
    for(int i=0; i<n*m; i++){
        J[i] = I_norm[i] + AWGN_generator();
    }

    toTXT(I_norm,"normCuda.txt",n,m);
    toTXT(J,"JCuda.txt",n,m);

    allPatches = makePatches(J,n,m,allPatches,patchSizeH,patchSizeW);

    //pass all the elements of 'allPatches' to 'P' array(NxpSize)
    P = (float*)malloc(N*pSize*sizeof(float));
    for(int i=0; i<N; i++){
        for(int j=0; j<pSize; j++){
            P[pSize*i+j] = allPatches[i].patchArray[j];
        }
    }

    cudaMemcpy(dev_P, P, N*sizePatch, cudaMemcpyHostToDevice);
    
    gauss = computeG_a(patchSizeH, patchSizeW, patchSigma);
    cudaMemcpy(dev_gauss, gauss, sizePatch, cudaMemcpyHostToDevice);

    float *patch_i = (float*)malloc(sizePatch);
    float *dev_patchI;
    cudaMalloc(&dev_patchI, sizePatch);
    float *wi_j = (float*)malloc(N*sizeof(float));
    float *dev_wij;
    cudaMalloc(&dev_wij, N*sizeof(float));

    tStart = tic();

    for(int i=0; i<N; i++){ 
        for(int j=0; j<pSize; j++){ 
            patch_i[j] = P[pSize*i +j];
        }
        cudaMemcpy(dev_patchI, patch_i, sizePatch, cudaMemcpyHostToDevice);
        dist<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dev_wij, dev_patchI,dev_P, dev_gauss,pSize,filtSigma);
        cudaMemcpy(wi_j, dev_wij, size, cudaMemcpyDeviceToHost);
        for(int j=0; j<N;j++){
            Z[i] += wi_j[j]; 
            w[N*i+j] = wi_j[j];
        }
    }
    cudaFree(dev_patchI);
    cudaFree(dev_gauss);
    
    float *dev_Z;
    cudaMalloc(&dev_Z, sizeof(float));
    for(int i=0; i<N; i++){
        for(int j=0; j<N;j++){
            wi_j[j] = w[N*i+j];
        }
        cudaMemcpy(dev_wij, wi_j,size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Z, &Z[i], sizeof(float), cudaMemcpyHostToDevice);
        dim<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dev_wij,dev_Z);
        cudaMemcpy(wi_j, dev_wij, size,cudaMemcpyDeviceToHost);
        for(int j=0; j<N;j++){
            w[N*i+j] = wi_j[j];
            If[i] += w[N*i+j] * J[j];
        }
    }
    double time = toc(tStart);

    toTXT(If,"IfCuda.txt",n,m);
    
    float* Dif = (float*)malloc(N*sizeof(float));
    for(int i=0; i<N; i++){
        Dif[i] =If[i] - J[i];
    }  

    toTXT(Dif,"DifCuda.txt",n,m);

    printf("Time: %f sec", time);    

    cudaFree(dev_I); cudaFree(dev_I_norm); cudaFree(dev_J); cudaFree(dev_P); cudaFree(dev_wij); cudaFree(dev_Z); 
    free(I); free(I_norm); free(J); free(patch_i); free(gauss); free(wi_j); free(Z); free(If); free(Dif); free(w);
    free(allPatches->patchArray);
    free(allPatches);
    return 0;
}