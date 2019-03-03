/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northestern University.
* All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/


/*
* $Created by: YIN FEI (email: ) 2019-01-22
*/

#include <math.h>
#include "LinerSoftmax.h"
#include "LogSoftmax.cuh"
#include "../XName.h"
#include "../XUtility.h"
#include "../core/reduce/ReduceSum.h"
#include "../core/reduce/ReduceMax.h"
#include "../core/movement/CopyValues.h"
#include "../../tensor/core/getandset/ConvertDataType.h"

using namespace nts;


namespace nts { // namespace nts(NiuTrans.Tensor)

	/* set TAYLOR number to limit the precision of taylor expansion */
	void set_TAYLOR(int x) {
		if (x >= 6)
			x = 6;
		TAYLOR = x;
	}

	/*
	log scale softmax y = log(e^x / \sum_{i} e^{x_i})
	>> x - input vector
	>> y - result
	>> leadDim - leading dimension (along which we perform reduction)
	*/
	void _LinerSoftmax(const XTensor * x, XTensor * y, int leadDim)
	{
		CheckNTErrors(!x->isSparse && !y->isSparse, "TODO!");
		CheckNTErrors(x && y, "Empty input tensors!");
		CheckNTErrors(x->dataType == X_INT, "not X_INT data type!");
	
		if (leadDim < 0)
			leadDim = x->order - 1;

		if (y->dimSize[leadDim] == 1) {
			y->SetZeroAll();
			return;
		}

		int leadDimRDI = x->order - leadDim - 1;

		int * dimSize = new int[x->order - 1];
		for (int i = 0; i < x->order; i++) {
			if (i < leadDim)
				dimSize[i] = -x->dimSize[i];
			else if (i > leadDim)
				dimSize[i - 1] = -x->dimSize[i];
		}


		XMem * mem = x->mem;
		XTensor * max = NULL;
		XTensor * sum = NULL;
		XTensor * blockx = NULL;
		XTensor * blocky = NULL;
		XTensor * blockMax = NULL;
		XTensor * blockSum = NULL;

		int dimensionSize = y->dimSizeRDI[leadDimRDI];
		int stride = 1;
		int blockSize = 1;
		int blockNum = 1;

		for (int i = 0; i < leadDimRDI; i++)
			stride *= y->dimSizeRDI[i];
		blockSize = stride * dimensionSize;
		blockNum = y->unitNum / blockSize;

		max = NewTensorBuf(x->order - 1, dimSize, x->dataType, x->denseRatio, x->devID, mem);
		sum = NewTensorBuf(x->order - 1, dimSize, x->dataType, x->denseRatio, x->devID, mem);

		/* finish in one days */
		_ReduceMax(x, max, leadDim);
		//_ReduceSum(x, sum, leadDim, max);
	
		if (x->devID >= 0) {
			if (leadDimRDI == 0) {
				blockSize = y->unitNum;
				blockNum = 1;
				blockx = NewTensor2D(blockSize / dimensionSize, -dimensionSize, x->dataType, x->devID, mem);
				blocky = NewTensor2D(blockSize / dimensionSize, -dimensionSize, x->dataType, x->devID, mem);
				blockMax = NewTensor2D(blockSize / dimensionSize, -1, x->dataType, x->devID, mem);
				blockSum = NewTensor2D(blockSize / dimensionSize, -1, x->dataType, x->devID, mem);
			}
			else {
				blockx = NewTensor2D(-stride, dimensionSize, x->dataType, x->devID, mem);
				blocky = NewTensor2D(-stride, dimensionSize, x->dataType, x->devID, mem);
				blockMax = NewTensor2D(-stride, 1, x->dataType, x->devID, mem);
				blockSum = NewTensor2D(-stride, 1, x->dataType, x->devID, mem);
			}
		}

		for (int k = 0; k < blockNum; k++) {
			/* walking distance */
			int m = stride;
			int n = dimensionSize;
			
			if (x->devID < 0) {
				int * ip = (int*)x->data + k * blockSize;
				int * op = (int*)y->data + k * blockSize;
				int * mp = (int*)max->data + k * blockSize / dimensionSize;
				int * sp = (int*)sum->data + k * blockSize / dimensionSize;
				
				for (int j = 0; j < m; j++) {
					int sumValue = sp[j];
					if (sumValue == 0) {
						for (int i = 0; i < n; i++)
							op[i * m + j] = 0;
					}
					else {
						for (int i = 0; i < n; i++) {
							// int r = (int)log(exp(ip[i * m + j] - mp[j]) / sp[j]);
							int r = 0;
							int toMul = 1;
							int temp = ip[i * m + j] - mp[j];

							for (int in = 0; in < TAYLOR; ++in) {
								r += toMul / DENOMINATOR[in];
								toMul *= temp;
							}

							/* original max line is -20 
							因为集合总是映射到原来的集合中
							即（-20,0）->（-20,0）
							现在由于是把所有的值放大了若干倍
							所以这里下线也要进行放大x倍
							下线预估放大的是100倍
							*/
							op[i * m + j] = MAX(r, -2000000);
							/* 这个线有问题 */
						}
					}
				}
			}
			else {
				/* use in gpu */
				/* liner : only consider the int type */
				/* omit the if to judge whether the type is X_INT */
				int * ip = (int*)x->data + k * blockSize;
				int * op = (int*)y->data + k * blockSize;
				int * mp = (int*)max->data + k * blockSize / dimensionSize;
				int * sp = (int*)sum->data + k * blockSize / dimensionSize;

				blockx->data = ip;
				blocky->data = op;
				blockMax->data = mp;
				blockSum->data = sp;
				

#ifdef USE_CUDA
				if (leadDimRDI == 0)
					_CudaLogSoftmaxSumMax(blockx, blocky, 1, blockSum, blockMax);
				else
					_CudaLogSoftmaxSumMax(blockx, blocky, leadDim, blockSum, blockMax);
#else
				ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
				blockx->data = NULL;
				blocky->data = NULL;
				blockMax->data = NULL;
				blockSum->data = NULL;
			}
		}

		DelTensorBuf(max);
		DelTensorBuf(sum);

		if (x->devID >= 0) {
			delete blockx;
			delete blocky;
			delete blockMax;
			delete blockSum;
		}

		delete[] dimSize;
	}

	/*
	log scale softmax y = log(e^x / \sum_{i} e^{x_i}) (return an XTensor structure)
	make a new tensor to keep the result and return it

	>> x - input vector
	>> leadDim - leading dimension (along which we perform reduction)
	<< return - y
	*/
	XTensor LinerSoftmax(const XTensor &x, int leadDim)
	{
		if (leadDim < 0)
			leadDim = x.order - 1;

		XTensor y(&x);
		y.SetTMPFlag();

		/* call _LogSoftmax function */
		_LinerSoftmax(&x, &y, leadDim);

		/* tensor connection */
		XLink::MakeLink(&x, NULL, &y, FUNC_LOGSOFTMAX);
		XLink::AddParamToHeadInt(&y, leadDim);

		return y;
	}

	/*
	log scale softmax y = log(e^x / \sum_{i} e^{x_i})
	make a new tensor to keep the result and return it

	>> x - input vector
	>> y - output vector
	>> leadDim - leading dimension (along which we perform reduction)
	*/
	void LinerSoftmax(const XTensor &x, XTensor &y, int leadDim)
	{
		if (!XTensor::IsSameShaped(&x, &y))
			InitTensor(&y, &x);

		/* call _LogSoftmax function */
		_LinerSoftmax(&x, &y, leadDim);

		/* tensor connection */
		XLink::MakeLink(&x, NULL, &y, FUNC_LOGSOFTMAX);
		XLink::AddParamToHeadInt(&y, leadDim);
	}

	/*
	log scale softmax y = log(e^x / \sum_{i} e^{x_i})
	>> x - input vector
	>> y - result
	>> leadDim - leading dimension (along which we perform reduction)
	*/
	void _FloatTestSoftmax(const XTensor * x, XTensor * y, int leadDim)
	{
		CheckNTErrors(!x->isSparse && !y->isSparse, "TODO!");
		CheckNTErrors(x && y, "Empty input tensors!");

		if (leadDim < 0)
			leadDim = x->order - 1;

		if (y->dimSize[leadDim] == 1) {
			y->SetZeroAll();
			return;
		}

		int leadDimRDI = x->order - leadDim - 1;
		int * dimSize = new int[x->order - 1];
		for (int i = 0; i < x->order; i++) {
			if (i < leadDim)
				dimSize[i] = -x->dimSize[i];
			else if (i > leadDim)
				dimSize[i - 1] = -x->dimSize[i];
		}

		XMem * mem = x->mem;
		XTensor * max = NULL;
		XTensor * sum = NULL;
		XTensor * blockx = NULL;
		XTensor * blocky = NULL;
		XTensor * blockMax = NULL;
		XTensor * blockSum = NULL;

		int dimensionSize = y->dimSizeRDI[leadDimRDI];
		int stride = 1;
		int blockSize = 1;
		int blockNum = 1;

		for (int i = 0; i < leadDimRDI; i++)
			stride *= y->dimSizeRDI[i];
		blockSize = stride * dimensionSize;
		blockNum = y->unitNum / blockSize;

		max = NewTensorBuf(x->order - 1, dimSize, x->dataType, x->denseRatio, x->devID, mem);
		sum = NewTensorBuf(x->order - 1, dimSize, x->dataType, x->denseRatio, x->devID, mem);

		_ReduceMax(x, max, leadDim);
		/* change one */
		_ReduceSumFloatTest(x, sum, leadDim, max, 1.0F, true);
		//_ReduceSum(x, sum, leadDim, max, 1.0F, true);

		
		if (x->devID >= 0) {
			if (leadDimRDI == 0) {
				blockSize = y->unitNum;
				blockNum = 1;
				blockx = NewTensor2D(blockSize / dimensionSize, -dimensionSize, x->dataType, x->devID, mem);
				blocky = NewTensor2D(blockSize / dimensionSize, -dimensionSize, x->dataType, x->devID, mem);
				blockMax = NewTensor2D(blockSize / dimensionSize, -1, x->dataType, x->devID, mem);
				blockSum = NewTensor2D(blockSize / dimensionSize, -1, x->dataType, x->devID, mem);
			}
			else {
				blockx = NewTensor2D(-stride, dimensionSize, x->dataType, x->devID, mem);
				blocky = NewTensor2D(-stride, dimensionSize, x->dataType, x->devID, mem);
				blockMax = NewTensor2D(-stride, 1, x->dataType, x->devID, mem);
				blockSum = NewTensor2D(-stride, 1, x->dataType, x->devID, mem);
			}
		}

		for (int k = 0; k < blockNum; k++) {
			int m = stride;
			int n = dimensionSize;

			if (x->devID < 0) {
				DTYPE * ip = (DTYPE*)x->data + k * blockSize;
				DTYPE * op = (DTYPE*)y->data + k * blockSize;
				DTYPE * mp = (DTYPE*)max->data + k * blockSize / dimensionSize;
				DTYPE * sp = (DTYPE*)sum->data + k * blockSize / dimensionSize;
				for (int j = 0; j < m; j++) {
					DTYPE sumValue = sp[j];
					if (sumValue == 0) {
						for (int i = 0; i < n; i++)
							op[i * m + j] = 0;
					}
					else {
						for (int i = 0; i < n; i++) {
							DTYPE r = (DTYPE)log(exp(ip[i * m + j] - mp[j]) / sp[j]);
							if (IsNAN(r))
								r = LOGPROB_MIN;
							if (IsINF(r))
								r = LOGPROB_MIN;

							op[i * m + j] = MAX(r, LOGPROB_MIN);
						}
					}
				}
			}
			else {
				DTYPE * ip = (DTYPE*)x->data + k * blockSize;
				DTYPE * op = (DTYPE*)y->data + k * blockSize;
				DTYPE * mp = (DTYPE*)max->data + k * blockSize / dimensionSize;
				DTYPE * sp = (DTYPE*)sum->data + k * blockSize / dimensionSize;

				blockx->data = ip;
				blocky->data = op;
				blockMax->data = mp;
				blockSum->data = sp;
#ifdef USE_CUDA
				if (leadDimRDI == 0)
				    /* change two */
					_CudaLogSoftmaxSumMaxFloatTest(blockx, blocky, 1, blockSum, blockMax);
				else
					_CudaLogSoftmaxSumMaxFloatTest(blockx, blocky, leadDim, blockSum, blockMax);
#else
				ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
				blockx->data = NULL;
				blocky->data = NULL;
				blockMax->data = NULL;
				blockSum->data = NULL;
			}
		}

		DelTensorBuf(max);
		DelTensorBuf(sum);

		if (x->devID >= 0) {
			delete blockx;
			delete blocky;
			delete blockMax;
			delete blockSum;
		}

		delete[] dimSize;
	}

} // namespace nts(NiuTrans.Tensor)
  
