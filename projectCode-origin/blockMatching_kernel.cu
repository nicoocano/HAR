#pragma once
#ifndef BLOCK_MATCHING_KERNEL
#define BLOCK_MATCHING_KERNEL
#define INDXs(s,i,j)   ((s) * (i) + (j) + 0)

__device__ double computeMatchKernel(unsigned char *im,
		    int im_step,
		    unsigned char *bl,
		    int bl_step,
		    int bl_cols,
		    int bl_rows,
		    int oi, 
		    int oj, 
		    int stride){
  
  if (!im || !bl) return 0.0;

  double nb = (bl_cols*bl_rows);
  double x = 0;
  for(int i = 0;i < bl_rows-stride+1;i+= stride){
    for(int j = 0;j < bl_cols-stride+1;j+= stride){
      unsigned char v1 = im[INDXs(im_step,oi+i,oj+j)];
      unsigned char v2 = bl[INDXs(bl_step,i,j)];
      x += (v2-v1)*(v2-v1);
      //im[INDXs(im_step,oi+i,oj+j)] = ABS(v2-v1);
    }
  }
  x = x / nb;
  //  printf("%f\n",x);
  return x;
}

struct DataOut{
	double minVal;
	int coord_i_min;
	int coord_j_min;
};

__global__ void blockMatching_kernel(int jend,int stride,unsigned char* im, int im_step, unsigned char *bl, int bl_step,int bl_cols,int bl_rows, DataOut* result){
		
	__shared__ DataOut tab_data_out[1024];
	
	DataOut temp;
	temp.minVal=1000000000000000;// a changer	
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	for(int j = 0;j < jend-stride+1;j+=stride){
	  double x = computeMatchKernel(im,im_step,
				  bl,bl_step,bl_cols,bl_rows,
				  tid,j,stride);
	  
		 if(x<temp.minVal){
			 temp.minVal=x;
			 temp.coord_i_min=tid;
			 temp.coord_j_min=j;
		}
	}
			
	tab_data_out[tid]=temp;
	__syncthreads();
	
	for (unsigned int s = 1; s<blockDim.x * blockDim.y; s *=2){
		int index = 2 * s * threadIdx.x;
		
		if (index < blockDim.x * blockDim.y){

			if (tab_data_out[index].minVal > tab_data_out[index + s].minVal){
				if((index + s) < blockDim.x * blockDim.y){
					tab_data_out[index] = tab_data_out[index + s];			
				}	
			}
		}
		__syncthreads();
	}

	if(threadIdx.x==0){
		result[blockIdx.x] = tab_data_out[0];
	}


	
	  
}
#endif
