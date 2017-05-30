__kernel void reduce_add(__global float* A, __local float* scratch, int stride, int size) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//If current element is beyond input size, set to 0 (padding)
	//Otherwise load into local memory
	scratch[lid] = (id >= size) ?  0 : A[id * stride];
	barrier(CLK_LOCAL_MEM_FENCE);


	//Sequential addressing
	for (; N > 0;) {

		//Odd block size detection for last unpaired element
		//Find new N value
		bool isOdd = false;
		if (N % 2 != 0) {
			isOdd = true;
			N = (N - 1) / 2;
		}
		else {
			N /= 2;
		}

		if (lid < N) {
			scratch[lid] += scratch[lid + N];
		}
		//If there is an unpaired element at the end of the block, add it to the first local
		if ((!lid && isOdd && N >= 1) && (!((id >= size) || (N * 2 >= size)))){
			scratch[lid] += scratch[N * 2];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		//0th element becomes block total sum
		A[id * stride] = scratch[lid];
	}
}

__kernel void histogram(__global float* I, __global int* O, float offset) {
	int id = get_global_id(0);
	float value = ((I[id] + offset) * 10);
	int newval = (int)(value + 1);
	newval--;
	//printf("%s\n", (I[id] * 10.0) + (offset * 10.0));
	/*
	int index;
	printf("Value: %f\n", value);
	float one = 1.0;
	if (value == one) {
		index = 1;
	} else {
		index = (int)value;
	}*/

	//printf("I[id]: %f, offset: %f, moves to O[%d]\n", I[id], offset ,value);
	//printf("Bin++: %d id: %d\n", (int)((I[id] + offset) * 10), id);
	atom_inc(&O[newval]);
}

__kernel void reduce_max(__global float* A, __local float* scratch, int stride, int size) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//If current element is beyond input size, set to 0 (padding)
	//Otherwise load into local memory
	scratch[lid] = (id >= size) ?  0 : A[id * stride];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	//Sequential addressing
	for (; N > 0;) {

		//Odd block size detection for last unpaired element
		//Find new N value
		bool isOdd = false;
		if (N % 2 != 0) {
			isOdd = true;
			N = (N - 1) / 2;
		}
		else {
			N /= 2;
		}
		if ((lid < N) && (scratch[lid + N] > scratch[lid])) {
			scratch[lid] = scratch[lid + N];
		}
		//If there is an unpaired element at the end of the block, compare with the first local
		if ((!lid && isOdd && N >= 1) && (!((id >= size) || (N * 2 >= size)))){
			if (scratch[lid + N] > scratch[lid]){
				scratch[lid] = scratch[N*2];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		//0th element becomes block max
		A[id * stride] = scratch[lid];
	}
}

__kernel void reduce_min(__global float* A, __local float* scratch, int stride, int size) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//If current element is beyond input size, set to 0 (padding)
	//Otherwise load into local memory
	scratch[lid] = (id >= size) ?  0 : A[id * stride];
	barrier(CLK_LOCAL_MEM_FENCE);

	//Sequential addressing
	for (; N > 0;) {
		//Odd block size detection for last unpaired element
		//Find new N value
		bool isOdd = false;
		if (N % 2 != 0) {
			isOdd = true;
			N = (N - 1) / 2;
		}
		else {
			N /= 2;
		}
		if ((lid < N) && (scratch[lid + N] < scratch[lid])) {
			scratch[lid] = scratch[lid + N];
		}
		//If there is an unpaired element at the end of the block, compare with the first local
		if ((!lid && isOdd && N >= 1) && (!((id >= size) || (N * 2 >= size)))){
			if (scratch[lid + N] < scratch[lid]){
				scratch[lid] = scratch[N*2];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		//0th element becomes block min
		A[id * stride] = scratch[lid];
	}
}

__kernel void diff_squared(__global float* A, const float val) {
	int id = get_global_id(0);
	float diff = A[id] - val;
	A[id] = diff * diff;
}