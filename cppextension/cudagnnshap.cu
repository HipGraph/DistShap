// This is GPU sampler of GNNShap.
// It is compiled as a shared library during the first run and called by the main GNNShap code.


#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#include <iostream>
#include <math.h>
#include <chrono>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>

#include <tuple>


#include <pybind11/pybind11.h>
#include <Python.h>

using namespace torch::indexing;
namespace py = pybind11;


// ######################DEVICE FUNCTIONS#########################


/* combination generation algorithm on cuda from
https://forums.developer.nvidia.com/t/enumerating-combinations/19980/4
*/

//calculate binomial coefficient
__inline__ __host__ __device__ unsigned int BinCoef(int n, int r) {
    unsigned int b = 1;  
    for(int i=0;i<=(r-1);i++) {
        b= (b * (n-i))/(i+1);
    }
    return(b);

    //the following is slower on CPU. I didn't test on GPU.
    //lround(std::exp( std::lgamma(n+1)-std::lgamma(n-k+1)-std::lgamma(k+1)));
}

//assigns the rth combination of n choose k to the array maskMat's row
__device__ int rthComb(unsigned int r, bool* rowPtr, bool* symRowPtr, int n,
  int k) { 
    int x = 1;  
    unsigned int y;
    for(int i=1; i <= k; i++) {
        y = BinCoef(n-x,k-i);
        while (y <= r) {
            r = r - y;
            x = x+1;
	        if (x > n)
		        return 0;
            y= BinCoef(n-x,k-i);
        }
        rowPtr[x-1] = true;
        symRowPtr[x-1] = false;
        x = x + 1;
    }
    return 1;

}


// ######################Cuda Kernels#########################

__global__ void kernelSampleGenerator(int nPlayers, int nHalfSamp,
  int* sizeLookup, bool* maskMat, int rndStartInd, int* devStartInds,
  int* devShuffleArr) {
  /*
  Cuda kernel code that samples symmetric pairs of coalitions. Symmetric pairs
  are added to second half of the mask matrix. For example, if we have 4
  samples, 1,3 and 2,4 are symmetric pairs.

  Parameters:
  nPlayers: int, number of players
  nHalfSamp: int, half number of samples to generate. We will get the symmetric
    pairs.
  sizeLookup: int*, array of coalition sizes
  maskMat: bool*, mask matrix to store the samples
  rndStartInd: int, start index of the random samples
  devStartInds: int*, device array of start indices
  devShuffleArr: int*, device array to store the shuffled indices

  Returns:
  None
  
  */

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nTotalThreads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
  int chunk = nHalfSamp/nTotalThreads + 1;

  if (tid*chunk >= nHalfSamp)
    return;

  int* localShuffleArr = devShuffleArr + tid * nPlayers;
  



  bool* mask = maskMat + chunk * tid * nPlayers;
  // symmetrics starts from the middle
  bool* maskSym = maskMat + nHalfSamp * nPlayers + chunk * tid * nPlayers; 
  int i, k;

  // nchoosek based sampling
  int fullCoalTaskEndInd = min(chunk*(tid + 1), rndStartInd);
  int rndTaskEndInd = min(chunk*(tid + 1), nHalfSamp);

  for (i = tid * chunk; i < fullCoalTaskEndInd; i++) {
    k  =  sizeLookup[i];
    //generate combination
    rthComb( i - devStartInds[k-1], mask, maskSym, nPlayers, k);
    mask += nPlayers; //move pointer to the next combination
    maskSym += nPlayers;
  }


  if (rndTaskEndInd <= fullCoalTaskEndInd)
    return;

  
  // random sampling
  // do random sampling here!

  curandState_t state;
  curand_init(1234, tid, 0, &state);
  int temp, y, z;
  for (z = 0; z < nPlayers; z++)
    localShuffleArr[z] = z;

  for (;i < rndTaskEndInd; i++) {
      //knuthShuffle Algorithm
      for (z = nPlayers - 1; z > 0; z--) {
            y = (int)(curand_uniform(&state)*(z + .999999));
            temp = localShuffleArr[z];
            localShuffleArr[z] = localShuffleArr[y];
            localShuffleArr[y] = temp;
      }
    
    for (int j = 0; j < sizeLookup[i]; j++){
      mask[localShuffleArr[j]] = true;
      maskSym[localShuffleArr[j]] = false;
    }

    mask += nPlayers; //move pointer to the next combination
    maskSym += nPlayers;
  }
}


__global__ void cudaFill(double* array, double value, int size) {
  /*

  Cuda kernel code to fill an array with a value. It is used to fill the kernel
  weights array.

  Parameters:
  array: double*, array to fill
  value: double, value to fill the array
  size: int, size of the array

  Returns:
  None
  */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotalThreads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    int chunk = size/nTotalThreads + 1;

    if (tid*chunk >= size)
        return;

    int end = min(chunk*(tid + 1), size);

    for (int i = tid * chunk; i < end; i++)
        array[i] = value;
}

__global__ void cudaFill(int* array, int value, int size) {
  /*
  Cuda kernel code to fill an array with a value. It is used to fill the size
  lookup array.

  Parameters:
  array: int*, array to fill
  value: int, value to fill the array
  size: int, size of the array

  Returns:
  None
  */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nTotalThreads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    int chunk = size/nTotalThreads + 1;

    if (tid*chunk >= size)
        return;

    int end = min(chunk*(tid + 1), size);

    for (int i = tid * chunk; i < end; i++)
        array[i] = value;
}


__global__ void kernelPartialNoShuffleArray(int nPlayers, int nHalfSamp,
  int nSelfLoops, int* sizeLookup, bool* maskMat, int rndStartIdx,
  int* startInds, int wNHalfSamp, int wOffset, unsigned int seed=1234) {
  /*
  Cuda kernel code that samples symmetric pairs of coalitions. Symmetric pairs
  are added to the next row of the original pairs. It can sample partially in
  multiple workers setting. If wOffset is 0 and wNHalfSamp is equal to
  nHalfSamp, then it will sample the whole samples.
  

  Parameters:
  nPlayers: int, number of players
  nHalfSamp: int, half number of samples to generate. We will get the
    symmetric pairs.
  sizeLookup: int*, array of coalition sizes
  maskMat: bool*, mask matrix to store the samples
  rndStartIdx: int, start index of the random samples
  startInds: int*, array of start indices for each coalition size
  devShuffleArr: int*, device array to store the shuffled indices
  wNHalfSamp: int, half number of samples for the current worker
  wOffset: int, offset for the current worker

  Returns:
  None
  */
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nTotalThreads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
  int chunk = wNHalfSamp/nTotalThreads + 1;

  if (tid*chunk >= wNHalfSamp)
    return;

  int rowSize = nPlayers + nSelfLoops;// row size for the mask matrix
  
  // long is to make sure there is no int overflow
  int64_t maskStartLoc = ((int64_t)chunk) * tid * rowSize * 2;
  
  // if (maskStartLoc < 0)
  //     printf("tid overflow: %lld", maskStartLoc);

  bool* mask = maskMat + maskStartLoc;
  // symmetric mask pointer. It is  assigned to the next row of the mask below.
  bool* maskSym;
  int i, k;

  
  //until next thread starts or the end of the worker samples
  int fullCoalEndIdx = min(min(rndStartIdx, wNHalfSamp + wOffset),
                          chunk*(tid + 1) + wOffset); 
  int rndEndIx = min(min(wNHalfSamp + wOffset, nHalfSamp),
                        chunk*(tid + 1) + wOffset);



  // nchoosek based sampling if in the full coalition range
  for (i = tid * chunk + wOffset; i < fullCoalEndIdx; i++) {
    k  =  sizeLookup[i - wOffset]; //coalition size
    maskSym = mask + rowSize;
    rthComb(i-startInds[k-1], mask, maskSym, nPlayers, k);//generate combination
    mask += rowSize * 2; //move pointer to the next combination
  }

  if (rndEndIx <= fullCoalEndIdx)
    return;

  
  // random sampling
  // do random sampling here!
  curandState_t state;
  curand_init(seed, tid, 0, &state);

  int j, rnd;
  
  for (;i < rndEndIx; i++) {
    j = 0;
    k = sizeLookup[i-wOffset];
    maskSym = mask + rowSize;

    while (j < k){
      rnd = curand(&state) % nPlayers;
      if (!mask[rnd]){
        mask[rnd] = true;
        maskSym[rnd] = false;
        j++;
      }
    }
    mask += rowSize * 2 ; //move pointer to the next combination
  }
}


// ######################HOST FUNCTIONS#########################

__inline__ double arraySum(double* arr, int n) {
  double sum = 0;
  for (int i = 0; i < n; i++)
    sum += arr[i];
  return sum;
}

__inline__ int arraySum(int* arr, int n) {
  int sum = 0;
  for (int i = 0; i < n; i++)
    sum += arr[i];
  return sum;
}

__inline__ void normalizeArray(double* arr, int n) {
  double sum = arraySum(arr, n);
  for (int i = 0; i < n; i++)
    arr[i] /= sum;
}

__inline__ void divideArray(double* arr, int n, double divisor) {
  for (int i = 0; i < n; i++)
    arr[i] /= divisor;
}

std::tuple<int, int> distributeSamplesToCoalSizes(int nPlayers, int nHalfSamp,
  double weightScale, int nSubsetSizes, int nPairedSubsetSizes, int* startInds,
  int *devSLookup, double *devKW, int wOffset, int wNHalfSamp)
{
  /*
  Distribute samples to the coalition sizes. The samples are distributed based
  on the coalition size. sizeLookup array contains the coalition size for each
  sample and startInds array contains the start index for each coalition size.
  The kernelWeights array is filled with the weights for each sample. The
  function returns the start index for the random samples and the number of full
  subset sizes.

  Note that sizeLookup, and kernelWeights are for the current worker only if
  the distributed explanations is used. It is unnecessary to compute for all
  samples, since we are only interested in the samples for the current worker.
  We need startInds if the worker has a fully enumerated subset size.

  Parameters:
  nPlayers: int, number of players
  nHalfSamp: int, half number of total samples
  weightScale: double, scale for the weights
  nSubsetSizes: int, number of subset sizes
  nPairedSubsetSizes: int, number of paired subset sizes
  startInds: int*, array to store the start indices for each coalition size
  devSLookup: int*, device array to store the coalition size for each sample
  devKW: double*, device array to store the kernel weights
  wOffset: int, offset for the current worker
  wNHalfSamp: int, half number of samples for the current worker

  Returns:
  rndStartIdx: int, start index for the random samples
  nFullSubsets: int, number of fully enumerated subset sizes
  */
  int nSamplesLeft = nHalfSamp; // number of samples left to distribute.
  long nSamplesUsed = 0; // number of samples used
  long sizeNSamples = 0; // number of samples for the current coalition size
  long wStart = wOffset; // worker sample start index
  long wEnd = min(wOffset + wNHalfSamp, nHalfSamp); // worker sample end index
  double sizeWeight; // weight for the current coalition size


  // size lookup array. create on the host, then copy to the device
  int* hostSLookup = new int[wNHalfSamp];
  int* tmpSLookPtr = hostSLookup; // size lookup pointer
  // double* hostKW = new double[wNHalfSamp * 2];
  double* tmpKWPointer = devKW;//hostKW; // kernel weights pointer

  long rangeSize;


  // weight vector to distribute samples
  double* weightVect = new double[nSubsetSizes];

  // compute weight vector
  for (int i = 1; i <= nSubsetSizes; i++) {
    weightVect[i-1] = exp(log(nPlayers - 1.0) - log(i) 
                      - log(nPlayers - i));
  }

  // we will get the symmetric except in the middle
  if (nSubsetSizes != nPairedSubsetSizes)
      weightVect[nPairedSubsetSizes] /= 2;

  // normalize weight vector to sum to 1
  normalizeArray(weightVect, nSubsetSizes);


  double * remWeightVect = new double[nSubsetSizes];
  std::copy(weightVect, weightVect + nSubsetSizes, remWeightVect);

  double sumKW = 0;
  startInds[0] = 0;

  // check if we have enough samples to iterate all coalitions for each subset.
  int nFullSubsets = 0;
  long nSubsets;
  for(int i = 1; i <= nSubsetSizes; i++){
    nSubsets =   BinCoef(nPlayers, i);//nChoosek(nPlayers, i);

    if (i > nPairedSubsetSizes){
      if (nSubsets % 2 != 0)
        std::cout << "Error: nSubsets is not even. Be careful!!!!" << std::endl;
      nSubsets /= 2;
      }
    if (nSamplesLeft * remWeightVect[i-1] + 1e-8 >= nSubsets){
      nFullSubsets++;
      // coalSizeNSamples[i-1] = nSubsets;
      nSamplesLeft -= nSubsets;
      startInds[i] = startInds[i-1] + nSubsets;

      sumKW += (weightScale*weightVect[i-1]);
      sizeWeight = (weightScale*weightVect[i-1]) / nSubsets;

      //check if the worker samples are in the current coalition size
      rangeSize = min(nSamplesUsed + nSubsets, wEnd) - max(nSamplesUsed, wStart);
      if(rangeSize > 0){
        //no need for more threads. We don't expect to be large for full subsets
        cudaFill<<<1, 64>>>(tmpKWPointer, sizeWeight, rangeSize * 2);
        tmpKWPointer += rangeSize * 2;
        std::fill(tmpSLookPtr, tmpSLookPtr + rangeSize, i);
        tmpSLookPtr += rangeSize;
      }

      nSamplesUsed += nSubsets;
      
      if (remWeightVect[i-1] < 1.0){
        divideArray(remWeightVect + i-1, nSubsetSizes -i+1,
                    1-remWeightVect[i-1]);
        // printMinMaxWeightVect(remWeightVect, nSubsetSizes);
      }

    }
    else{
      break;
    }    
  }

  // remaining samples get the same weight
  double remKw = (weightScale- sumKW)/nSamplesLeft;

  //check if the worker has samples in the remaining samples
  rangeSize = wEnd - max(nSamplesUsed, wStart);
  if (rangeSize > 0)
    // set them at once (same weight).
    cudaFill<<<4, 256>>>(tmpKWPointer, remKw, rangeSize * 2);
  
  int rndStartIdx = nHalfSamp - nSamplesLeft;

  // if we have enough samples to iterate all coalitions for each subset size,
  // then we are done.
  if (nFullSubsets != nSubsetSizes) {
    int remSamples = nSamplesLeft;
    // round up for the first size, then round down, then round up, etc.

    //distribute the remaining samples to the subset sizes
    for (int i = nFullSubsets; i < nSubsetSizes - 1; i++) {
      // extra check to avoid negative number of samples for the middle coal.
      // Might be redundant
      if (nSamplesLeft <= 0) {
        nSamplesLeft = 0;
        break;
      }

      sizeNSamples = min((int)ceil(remSamples * remWeightVect[i]),nSamplesLeft);
      
      nSamplesLeft -= sizeNSamples;


      rangeSize = min(nSamplesUsed + sizeNSamples, wEnd) 
                  - max(nSamplesUsed, wStart);
      if(rangeSize > 0){
        std::fill(tmpSLookPtr, tmpSLookPtr + rangeSize, i+1);
        tmpSLookPtr += rangeSize;

        // early exit if the worker samples are done
        if (nSamplesUsed + sizeNSamples >= wEnd){
          nSamplesUsed += sizeNSamples;
          break;
        }
      }

      nSamplesUsed += sizeNSamples;
    }
  }

  //add the remaining samples to the middle coal.
  // This might be used for the last worker
  if (nSamplesLeft> 0 &&  nSamplesUsed< wEnd){
    //make sure the correct range is used
    nSamplesUsed = max(nSamplesUsed, wStart);

    std::fill(tmpSLookPtr, tmpSLookPtr + (wEnd - nSamplesUsed), nSubsetSizes);
  }

  cudaMemcpy(devSLookup, hostSLookup, wNHalfSamp * sizeof(int),
            cudaMemcpyHostToDevice);

  delete[] hostSLookup;
  delete[] weightVect;
  delete[] remWeightVect;

  return std::make_tuple(rndStartIdx, nFullSubsets);

}


void printcudaMemoryErrorIfFails(cudaError_t err, size_t size,
  const char arrayName[]){

    if (err != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        // 1073741824.0 is 1024^3
        double requested_mem_gb = static_cast<double>(size) / 1073741824.0;
        double free_mem_gb = static_cast<double>(free_mem) / 1073741824.0;
        double total_mem_gb = static_cast<double>(total_mem)  / 1073741824.0;
        
        printf("CUDA error in allocating %s: Requested:%.2f GB, Available: %.2f GB, Total: %.2f GB\n",
          arrayName,requested_mem_gb, free_mem_gb, total_mem_gb);
        printf("CUDA error message: %s\n", cudaGetErrorString(err));
    }
}


void cudapartialAdjacentSymSample(torch::Tensor maskMatTensor,
  torch::Tensor kWTensor, int nPlayers, int nSamples, int wOffset,
  int nSelfLoops, int worldSize, int rank, double weightScale, int nBlocks, 
  int nThreads) {
  /*
  This function creates samples for the symmetric pairs of coalitions. Symmetric
  pairs are added to the next row of the original pairs. It can sample partially
  in multiple workers setting. If wOffset is 0 and wNHalfSamp is equal to
  nHalfSamp, then it will sample the whole samples.

  It doesn't use a shuffle arrays in the kernel. Hence, it uses less memory.
  Set worldSize=1, rank=0, wOffset=0 if you want to sample the whole samples.

  nselfLoops is used to allocate space for the self loops. It is not used in the
  sampling. The self loops are added to the end of the row. It is useful for the
  GNNShap algorithm. No need to allocate selfloop added mask later.

  Parameters:
  maskMatTensor: torch::Tensor, mask matrix to store the samples
  kWTensor: torch::Tensor, kernel weights
  nPlayers: int, number of players
  nSamples: int, number of samples to generate
  wOffset: int, offset for the current worker
  nSelfLoops: int, number of self loops
  worldSize: int, number of workers
  rank: int, rank of the current worker
  weightScale: double, scale for the weights
  nBlocks: int, number of blocks for cuda kernel
  nThreads: int, number of threads for cuda kernel

  Returns:
  None
  */

  int wNSamples = maskMatTensor.size(0);

  int wNHalfSamp = wNSamples/2; // half number of samples for the worker


  bool *devMaskMat = maskMatTensor.data_ptr<bool>();

  double *devKW = kWTensor.data_ptr<double>(); // kernel weights

  // set the even rows to 0, columns until nPlayers
  maskMatTensor.index_put_({Slice(0, None, 2), Slice(0, nPlayers)}, 0);

  //Assuming an even number of samples. Should be checked in the python side.
  int nHalfSamp = nSamples/2; // half number of total samples.

  
  int nSubsetSizes = ceil((nPlayers - 1) / 2.0); // number of subset sizes
  // coalition size in the middle not a paired subset
  // if nPlayers=4, 1 and 3 are pairs, 2 doesn't have a pair
  int nPairedSubsetSizes = floor((nPlayers - 1) / 2.0);


  int *devSLookup; // device size lookup. coalition size for each sample
  size_t size = wNHalfSamp * sizeof(int);
  
  cudaError_t err = cudaMalloc((void**)&devSLookup, size);
  
  printcudaMemoryErrorIfFails(err, size, "devSLookup");
  
  

  // coalition size sample start indices.
  int* startInds = new int[nSubsetSizes+1]; 

  //distribution of samples to coalition sizes
  int rndStartIdx, nFullSubsets;
  std::tie(rndStartIdx, nFullSubsets) = distributeSamplesToCoalSizes(
      nPlayers, nHalfSamp, weightScale/2, nSubsetSizes, nPairedSubsetSizes,
      startInds, devSLookup, devKW, wOffset, wNHalfSamp);



  int* devStartInds; // device start indices
  /* create & copy the start indices to the device if the worker has a fully
  enumerated subset size.*/
  if (rndStartIdx > wOffset){
    size = (nFullSubsets+1) * sizeof(int);
    cudaMalloc((void**)&devStartInds, size);
    err = cudaMemcpy(devStartInds, startInds, (nFullSubsets+1) * sizeof(int),
                    cudaMemcpyHostToDevice);
    printcudaMemoryErrorIfFails(err, size, "devStartInds");
  }
  

  int seed = time(NULL);
  kernelPartialNoShuffleArray<<<nBlocks, nThreads>>>(nPlayers, nHalfSamp,
    nSelfLoops, devSLookup, devMaskMat,rndStartIdx, devStartInds, wNHalfSamp,
    wOffset, seed);
  cudaDeviceSynchronize(); // wait for the kernel to finish
  
  

  delete[] startInds;
  cudaFree(devSLookup);
  
  if (rndStartIdx > wOffset)
    cudaFree(devStartInds);
  
}


void cudaSample(torch:: Tensor maskMatTensor, torch::Tensor kWTensor,
  int nPlayers, int nSamples, int nBlocks = 1, int nThreads = 6) {
  /*
  Cuda sampler function that generates symmetric pairs of coalitions. Symmetric
  pairs are added to the second half of the mask matrix. For example, if we
  have 4 players, 1,3 and 2,4 are symmetric pairs. The function fills the mask
  matrix with the samples and kernel weights.


  Parameters:
  maskMatTensor: torch::Tensor, mask matrix to store the samples
  kWTensor: torch::Tensor, kernel weights
  nPlayers: int, number of players
  nSamples: int, number of samples to generate
  nBlocks: int, number of blocks for cuda kernel
  nThreads: int, number of threads for cuda kernel

  Returns:
  None
  */

  // we will get symmetric samples. no need to compute the other half
  int nHalfSamp = nSamples/2 ;
  int nSamplesLeft = nHalfSamp; 

  int* sizeLookup = new int[nSamplesLeft];
  double* kernelWeights = new double[nSamples];

  int* tmpSLookPtr = sizeLookup;
  double* tmpKWPointer = kernelWeights;

  
  int nSubsetSizes = ceil((nPlayers - 1) / 2.0); // number of subset sizes
  // coalition size in the middle not a paired subset
  // if nPlayers=4, 1 and 3 are pairs, 2 doesn't have a pair
  int nPairedSubsetSizes = floor((nPlayers - 1) / 2.0);

  // number of samples for each subset size
  int* coalSizeNSamples = new int[nSubsetSizes];

  // coalition size sample start indices
  int* startInds = new int[nSubsetSizes+1];
  // weight vector to distribute samples
  double* weightVect = new double[nSubsetSizes];


  // compute weight vector
  for (int i = 1; i <= nSubsetSizes; i++) {
    weightVect[i-1] = ((nPlayers - 1.0) / (i * (nPlayers - i)));
  }


  // we will get the symmetric except in the middle
  if (nSubsetSizes != nPairedSubsetSizes)
      weightVect[nPairedSubsetSizes] /= 2;

  // normalize weight vector to sum to 1
  normalizeArray(weightVect, nSubsetSizes);

  double * remWeightVect = new double[nSubsetSizes];
  std::copy(weightVect, weightVect + nSubsetSizes, remWeightVect);

  // std::cout << "initial remWeightVect: ";
  // for (int b = 0; b < nSubsetSizes; b++){
  //   std::cout << remWeightVect[b] << " ";
  // }
  // std::cout << std::endl;

  double sumKW = 0;
  startInds[0] = 0;

  // check if we have enough samples to iterate all coalitions for each subset.
  int nFullSubsets = 0;
  long nSubsets;
  for(int i = 1; i <= nSubsetSizes; i++){
    nSubsets =   BinCoef(nPlayers, i);//nChoosek(nPlayers, i);

    if (i > nPairedSubsetSizes){
      if (nSubsets % 2 != 0)
        std::cout << "Error: nSubsets is not even. Be careful!!!!" << std::endl;
      nSubsets /= 2;
      // std::cout << "inside if middle full sample control case" << std::endl;
      }

    if (nSamplesLeft * remWeightVect[i-1] + 1e-8 >= nSubsets){
      nFullSubsets++;
      coalSizeNSamples[i-1] = nSubsets;
      nSamplesLeft -= nSubsets;
      startInds[i] = startInds[i-1] + nSubsets;

      sumKW += (50*weightVect[i-1]);
      std::fill(tmpKWPointer, tmpKWPointer + nSubsets,
                (50*weightVect[i-1]) / nSubsets);
      std::fill(tmpSLookPtr, tmpSLookPtr + nSubsets, i);

      tmpKWPointer += nSubsets;
      tmpSLookPtr += nSubsets;
      
      if (remWeightVect[i-1] < 1.0){
        divideArray(remWeightVect + i-1, nSubsetSizes -i+1,
                    1-remWeightVect[i-1]);
      }

    }
    else{
      break;
    }    
  }

  // use this if we want equal weights for each randomly sampled coalitions.
  double remKw = (50.0 - sumKW)/nSamplesLeft;
  std::fill(tmpKWPointer, tmpKWPointer + nSamplesLeft, remKw);
  tmpKWPointer += nSamplesLeft;

  int rndStartInd = nHalfSamp - nSamplesLeft;

  // if we have enough samples to iterate all coalitions for each subset size,
  // then we are done.
  if (nFullSubsets != nSubsetSizes){
    int remSamples = nSamplesLeft;
    bool roundUp = true;
    for (int i = nFullSubsets; i < nSubsetSizes - 1; i++){
      
      // extra check to avoid negative number of samples for the middle coal.
      // Might be redundant
      if (nSamplesLeft <= 0) {
        nSamplesLeft = 0;
        break;
      }

      if (roundUp)
        coalSizeNSamples[i] = min((int)ceil(remSamples * remWeightVect[i]),
                                  nSamplesLeft);
      else
        coalSizeNSamples[i] = min((int)floor(remSamples * remWeightVect[i]),
                                  nSamplesLeft);
      nSamplesLeft -= coalSizeNSamples[i];

      /* if we want different weights for each randomly sampled coalition sizes,
      we can use this. However, experiments show that it doesn't make a 
      difference.
      std::fill(tmpKWPointer, tmpKWPointer + coalSizeNSamples[i],
                (50*weightVect[i]) / coalSizeNSamples[i]);
      tmpKWPointer += coalSizeNSamples[i];
      */

      std::fill(tmpSLookPtr, tmpSLookPtr + coalSizeNSamples[i], i+1);
      tmpSLookPtr += coalSizeNSamples[i];

      startInds[i+1] = startInds[i] + coalSizeNSamples[i];

      roundUp = !roundUp;
    }
    /* add the remaining samples to the middle coal. I removed the middle coal
    from the loop above to avoid negative number of samples for the middle coal.
    */
    coalSizeNSamples[nSubsetSizes-1] = nSamplesLeft;

    //startInds[nSubsetSizes-1] = startInds[nSubsetSizes-2] + nSamplesLeft;


    /* uncomment this if we want different weights for each randomly sampled
    coalition sizes. However, experiments show that it doesn't make a
    difference.
    std::fill(tmpKWPointer, tmpKWPointer + nSamplesLeft,
              (50*remWeightVect[nSubsetSizes-1]) / nSamplesLeft);
    tmpKWPointer += nSamplesLeft;
    */   
    
    std::fill(tmpSLookPtr, tmpSLookPtr + nSamplesLeft, nSubsetSizes);

    if (coalSizeNSamples[nSubsetSizes-1] < 0)
      std::cout << "Error: negative number of samples for the middle coalition"
                << std::endl;

    nSamplesLeft = 0;
  }

  // symmetric weights. No need to compute the other half, no need to flip
  memcpy(tmpKWPointer, kernelWeights, nHalfSamp * sizeof(double));

  bool *devMaskMat = maskMatTensor.data_ptr<bool>();
  // cudaMalloc(&devMaskMat, nSamples * nPlayers * sizeof(bool));
  // cudaMemset(devMaskMat, false, nHalfSamp * nPlayers * sizeof(bool));
  cudaMemset(devMaskMat + nPlayers * nHalfSamp, true,
            nHalfSamp * nPlayers * sizeof(bool));

  int *deviceSizeLookup;
  cudaMalloc(&deviceSizeLookup, nHalfSamp * sizeof(int));
  cudaMemcpy(deviceSizeLookup, sizeLookup, nHalfSamp * sizeof(int),
            cudaMemcpyHostToDevice);


  int* devShuffleArr;
  cudaMalloc(&devShuffleArr, nBlocks * nThreads * nPlayers * sizeof(int));

  int* devStartInds; // device start indices
  cudaMalloc(&devStartInds, nSubsetSizes * sizeof(int));
  cudaMemcpy(devStartInds, startInds, nSubsetSizes * sizeof(int),
            cudaMemcpyHostToDevice);

  kernelSampleGenerator<<<nBlocks, nThreads>>>(nPlayers, nHalfSamp, 
      deviceSizeLookup, devMaskMat, rndStartInd, devStartInds, devShuffleArr);

  
  cudaMemcpy(kWTensor.data_ptr<double>(), kernelWeights,
            nSamples * sizeof(double), cudaMemcpyHostToDevice);
  
  cudaDeviceSynchronize();

  free(sizeLookup);
  free(kernelWeights);
  free(coalSizeNSamples);
  free(startInds);
  free(weightVect);
  free(remWeightVect);
  cudaFree(deviceSizeLookup);
  cudaFree(devStartInds);
  cudaFree(devShuffleArr);
}




// ######################PYTHON BINDINGS#########################

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample", &cudaSample, "Cuda Sample");
  m.def("partialAdjacentSymSample", &cudapartialAdjacentSymSample,
        "Cuda Partial Adjacent Symmetric Sample");
}
