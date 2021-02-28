#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#define PI 3.1415926536
#define e 2.718281828459

struct timeval tic(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv;
}

typedef struct Patches
{
    int index;
    double central;
    double* patchArray;
}Patch;

typedef struct Vars{
    double* wi_j;
}Var;

double toc(struct timeval begin){
    struct timeval end;
    gettimeofday(&end,NULL);
    double stime = ((double)(end.tv_sec-begin.tv_sec)*1000)+((double)(end.tv_usec-begin.tv_usec)/1000);
    stime = stime / 1000;
    return (stime);
}

double* readFile(int n, int m, char *file_path){
    FILE* ptrFile = fopen(file_path, "r");

    double *I = (double*)malloc(n*m*sizeof(double));

    if (!ptrFile){
        printf("Error Reading File\n");
        exit (0);
    }

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            fscanf(ptrFile,"%lf,", &I[n*i+j]);
        }
    }

    fclose(ptrFile);

    return I;
}

void toTXT(double* array,char *output, int n, int m){
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

double* normalization(double* I, int n, int m){
    double min = INFINITY;
    double max = -1;
    double* I_normalized = (double*)malloc(n*m*sizeof(double));

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            if(I[n*j+i]<min) min= I[n*j+i];
        }
    }

    for(int i=0; i<n*m; i++){
        if((I[i]-min)>max) max = I[i]-min;
    }

    for(int i=0; i<n*m; i++){
        I_normalized[i] = (I[i] - min) / max ;
    }

    return I_normalized;
}


double AWGN_generator()     //https://www.embeddedrelated.com/showcode/311.php
{/* Generates additive white Gaussian Noise samples with zero mean and a standard deviation of 1. */
  double dev = 0.03162; //var = 0.01
  double temp1;
  double temp2;
  double result;
  int p = 1;

  while( p > 0 )
  {
	temp2 = ( rand() / ( (double)RAND_MAX ) ); /*  rand() function generates an
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

  temp1 = cos( ( 2.0 * (double)PI ) * rand() / ( (double)RAND_MAX ) );
  result = sqrt( -2.0 * log( temp2 ) ) * temp1;

  return result * dev;	// return the generated random sample to the caller

}// end AWGN_generator()




Patch* makePatches(double* J, int n, int m, Patch* allPatches, int patchSizeH, int patchSizeW){
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

double* computeG_a(int patchSizeH, int patchSizeW, double patchSigma){
    double* gauss = (double*)malloc(patchSizeH*patchSizeW*sizeof(double));

    for (int i = 0; i < patchSizeH; i++) {
        for (int j = 0; j < patchSizeW; j++) {
            double y = i - (patchSizeH - 1) / 2.0;
            double x = j - (patchSizeW - 1) / 2.0;
            gauss[patchSizeW*i+j] = (1/2.0) * exp(-(x * x + y * y) / (2.0 * PI * patchSigma * patchSigma));
        }
    }    
    return gauss;
}

double EuclideanDistance(double* A, double* B, double* V, int patchSizeH, int patchSizeW){
    double dist = 0;

    for (int i = 0; i < patchSizeH; i++) {
       for (int j = 0; j < patchSizeW; j++) {
            dist += V[patchSizeH*i+j] * pow(A[patchSizeH*i+j] - B[patchSizeH*i+j],2);
       }
    }
    dist = sqrt(dist);

    return dist;
}

int main(int argc, char *argv[]){
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int patchSizeH = atoi(argv[3]);
    int patchSizeW = atoi(argv[4]);
    double patchSigma =5/3;
    double filtSigma =0.01 ;

    char* file_path;
    file_path=(char*)malloc(strlen(argv[5])*sizeof(char));
    memcpy(file_path,argv[5],strlen(argv[5]));

    struct timeval tStart;
   
    double* I = (double*)malloc(n*m*sizeof(double));
    double* I_normalized = (double*)malloc(n*m*sizeof(double));
    double* J = (double*)malloc(n*m*sizeof(double));
    double* If = (double*)malloc(n*m*sizeof(double));
    
    Patch* allPatches;
    allPatches = (Patch*)malloc(n*m*sizeof(Patch));

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            allPatches[n*j+i].patchArray = (double*)malloc(patchSizeH*patchSizeW*sizeof(double));
        }
    }    
  
    double* gauss = (double*)malloc(patchSizeH*patchSizeW*sizeof(double));
    double* Z = (double*)malloc(n*m*sizeof(double));

    Var* w;
    w = (Var*)malloc(n*m*sizeof(Var));
    
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            w[n*j+i].wi_j = (double*)malloc(n*m*sizeof(double));
        }
    }    

    I = readFile(n,m,file_path);

    I_normalized = normalization(I,n,m);
    toTXT(I_normalized,"norm_v0.txt",n,m);

    for(int i=0; i<n*m; i++){
        J[i] = I_normalized[i] + AWGN_generator();
    }
    toTXT(J,"J_v0.txt",n,m);

    allPatches = makePatches(J,n,m,allPatches,patchSizeH,patchSizeW);    

    gauss = computeG_a(patchSizeH, patchSizeW, patchSigma);

    // for(int i=0; i<patchSizeH; i++){
    //     for(int j=0; j<patchSizeW; j++){
    //         printf("%lf  ", gauss[patchSizeH*j+i]);
    //     }
    //     printf("\n");
    // }

    tStart = tic();

    double d = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){          
            double* patch_i = allPatches[n*i+j].patchArray;
            
            for(int o=0; o<n; o++){
                for(int k=0; k<m; k++){
                    d = EuclideanDistance(patch_i, allPatches[n*o+k].patchArray,gauss,patchSizeH,patchSizeW);
                    w[n*i+j].wi_j[n*o+k] = exp(- pow(d,2)/filtSigma);
                    Z[n*i+j] += w[n*i+j].wi_j[n*o+k];
                } 
            }
        }
    }

    double* a = (double*)malloc(n*m*sizeof(double));
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            for(int o=0; o<n; o++){
                for(int k=0; k<m; k++){
                    w[m*i+j].wi_j[m*o+k] = w[m*i+j].wi_j[m*o+k] / Z[m*i+j];
                    //in order to check if a[n*i+j]
                    //a[n*i+j] += w[n*i+j].wi_j[n*o+k];
                    If[m*i+j] += w[m*i+j].wi_j[m*o+k] * J[m*o+k];
                }
            }
            //  printf("   %lf", a[n*i+j]);
            // printf("\n");
        }
    }

    double time = toc(tStart);

    // for(int i=0; i<n; i++){
    //     for(int j=0; j<m; j++){
    //         for(int o=0; o<n; o++){
    //             for(int k=0; k<m; k++){
    //                 If[m*i+j] += w[m*i+j].wi_j[m*o+k] * J[m*o+k];
    //             }
    //         }
    //     }
    // }
    toTXT(If,"If_v0.txt",n,m);

    double* Dif = (double*)malloc(n*m*sizeof(double));
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            Dif[m*i+j] =If[m*i+j] - J[m*i+j] ;
        }
    }
    toTXT(Dif,"Dif_v0.txt",n,m);
    
    printf("Time: %lf sec\n", time);

    free(I); free(I_normalized); free(J); free(If); free(Dif);
    free(allPatches->patchArray);
    free(allPatches);
    free(w->wi_j);
    free(w);
}
