#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

void parse_file(std::string filePath, std::vector<float>&dest) {

	//Open file
	std::ifstream file(filePath, std::ios::in | std::ios::binary);

	//If the file could be opened, parse
	if (file.is_open()) {
		
		//Scroll to the end to get the length of the file
		file.seekg(0, ios_base::end);
		int len = file.tellg();

		//Scroll back to the beginning ready to read
		file.seekg(0, ios_base::beg);

		//Read file into char array
		char* contents = new char[len];
		file.read(&contents[0], len);

		file.close();
		
		int spaces = 0;
		for (int i = 0; i < len; i++){
			if (contents[i] == ' ') {
				//Count spaces to identify last column
				spaces++;
			}

			//If at the last column
			if (spaces == 5) {
				spaces = 0; //Reset spaces for next count
				int size = 0;

				//Save array position ready for next iteration
				int position = i + 1;

				//Calculate the length of the number from the space to the end of the line
				do {
					i++;
					size++;
				} while (contents[i + 1] != '\r');

				//Temp char array for number
				char* number = new char[size + 1];

				//Fill up char array with entire column
				for (int c = 0; c < size; c++, position++) {
					number[c] = contents[position];
				}

				//Terminate string
				number[size] = '\0';

				//Convert to float and push to vector
				dest.push_back(atof(number));
			}
		}
	}
	else {
		//Faliure to open file
		cout << "Unable to open file " << filePath << std::endl;
		getchar();
	}

	
}

//Calculate the maximum factor of n that is less than the given max
int max_factor(int n, int top) {
	int d = 2;
	int maximum = 2;
	while (n > 1) {
		//While N is divisible by D
		while (n % d == 0) {
			//If this value is a larger n less than the max, use it
			if ((n > maximum) && (n <= top))
				maximum = n;
			n /= d;
		}
		d++;
	}
	return maximum;
}

//Calculate the squared differences from an input value on a vector
std::vector<float> diff_sq(std::vector<float> values, float sub,cl::Program* program, cl::Context* context, cl::CommandQueue* queue, cl::Device* device) {
	
	size_t input_size = values.size() * sizeof(float); //Size in bytes
	size_t input_elements = values.size(); //Number of elements

	//Single read/write buffer input
	cl::Buffer buffer(*context, CL_MEM_READ_WRITE, input_size);
	queue->enqueueWriteBuffer(buffer, CL_TRUE, 0, input_size, &values[0]);

	//Setup kernel
	cl::Kernel kernel = cl::Kernel(*program, "diff_squared");
	kernel.setArg(0, buffer);
	kernel.setArg(1, sub);

	//Profile, execute with max group size, read
	cl::Event prof_event;
	queue->enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(max_factor(input_elements, kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*device))), NULL, &prof_event);
	queue->enqueueReadBuffer(buffer, CL_TRUE, 0, input_size, &values[0]);

	//Display profiling information
	std::cout << "DSQ" << " execution [ns]: " <<
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " Full: " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
	return values;
}

//Parallel reduce on vector
float reduce(std::vector<float> values, cl::Kernel kernel, cl::Program* program , cl::Context* context, cl::CommandQueue* queue, cl::Device* device, string name){
	
	//Pre-pad vector into a multiple of 16
	size_t assumed_size = 16;
	size_t padding_size = values.size() % assumed_size;

	if (padding_size) {
		std::vector<int> A_ext(assumed_size - padding_size, 0);
		values.insert(values.end(), A_ext.begin(), A_ext.end());
	}


	size_t input_elements = values.size();//number of input elements
	size_t input_size = values.size() * sizeof(float);//size in bytes
	
	//Single buffer to read/write for re-use across iterations
	cl::Buffer buffer_reduce(*context, CL_MEM_READ_WRITE, input_size);
	queue->enqueueWriteBuffer(buffer_reduce, CL_TRUE, 0, input_size, &values[0]);
	kernel.setArg(0, buffer_reduce);

	//Maximum work group size
	int max_groups = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*device);
	
	//Iterate reductions until there is only 1 element in the stride
	for (int stride = 1, elements = input_elements; elements > 1; ) {
		cl::Event evt;

		//Calculate the maximum group size possible
		int group_size = max_factor(elements, max_groups);

		//Set arguments
		kernel.setArg(1, cl::Local(group_size * sizeof(float)));
		kernel.setArg(2, stride);
		kernel.setArg(3, elements);

		//Execute
		queue->enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(elements), cl::NDRange(group_size), NULL, &evt);
		
		//If there are an odd number of elements, round upwards with int division
		elements = (elements % (int)group_size) ? elements / (int)group_size + 1 : elements / (int)group_size;

		//Stride now steps over all elements just reduced
		stride *= (int)group_size;

		//Wait for event data
		evt.wait();

		//Display profiling information
		std::cout << name << " execution [ns]: " <<
			evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			evt.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " Full: " << GetFullProfilingInfo(evt, ProfilingResolution::PROF_US) << std::endl;	
	}

	//Read data from reductions
	queue->enqueueReadBuffer(buffer_reduce, CL_TRUE, 0, input_size, &values[0]);
	
	return values[0];
}

std::vector<int> bucket_sort(std::vector<float> values, float minimum, float maximum, cl::Program* program, cl::Context* context, cl::CommandQueue* queue, cl::Device* device, string name) {
	
	size_t input_elements = values.size();//number of input elements
	size_t input_size = values.size() * sizeof(float);//size in bytes

	//Get the range of the data to calculate the bin count (10 values for every decimal)
	float range = maximum - minimum;

	int bin_count = (int)(range * 10) + 1;
	std::vector<int> bins(bin_count);

	//Input buffer contains data, output buffer is the filled bins
	cl::Buffer buffer_input(*context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer buffer_output(*context, CL_MEM_READ_WRITE, bin_count * sizeof(int));

	//Write both buffers
	queue->enqueueWriteBuffer(buffer_input, CL_TRUE, 0, input_size, &values[0]);
	queue->enqueueWriteBuffer(buffer_output, CL_TRUE, 0, bin_count * sizeof(int), &bins[0]);

	//Profiling event
	cl::Event evt;

	//Set up kernel with the distance of the minimum from 0 as the offset
	cl::Kernel kernel = cl::Kernel(*program, "histogram");
	kernel.setArg(0, buffer_input);
	kernel.setArg(1, buffer_output);
	kernel.setArg(2, (minimum * -1.0f));
	
	//Execute then read
	queue->enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(max_factor(input_elements, kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*device))), NULL, &evt);
	queue->enqueueReadBuffer(buffer_output, CL_TRUE, 0, bin_count * sizeof(int), &bins[0]);

	//Wait for event data
	evt.wait();

	//Print execution time
	std::cout << name << " execution [ns]: " <<
		evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		evt.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " Full: " << GetFullProfilingInfo(evt, ProfilingResolution::PROF_US) << std::endl;

	//Return filled histogram
	return bins;
}

int main(int argc, char **argv) {
	int platform_id = 0;
	int device_id = 0;

	//Command line options
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//Load text file data
	vector<float> temps;
	parse_file("temp_lincolnshire.txt", temps);
	
	try {
		//Set up openCL objects
		cl::Context context = GetContext(platform_id, device_id);
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		//Queue with profiling
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		//Get device
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		//Load kernels
		cl::Program::Sources sources;
		AddSources(sources, "kernels.cl");
		cl::Program program(context, sources);

		//Build program
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		auto begin = std::chrono::high_resolution_clock::now();

		//Calculate sum and mean
		float sum = reduce(temps, cl::Kernel(program, "reduce_add"), &program, &context, &queue, &device, "Sum");
		float mean = sum / temps.size();

		//Calculate maximum and minimum
		float maximum = reduce(temps, cl::Kernel(program, "reduce_max"), &program, &context, &queue, &device, "Max");
		float minimum = reduce(temps, cl::Kernel(program, "reduce_min"), &program, &context, &queue, &device, "Min");

		//Calculate variance by reducing the output of the squared differences
		float variance = reduce(
			diff_sq(temps, mean, &program, &context, &queue, &device),
			cl::Kernel(program, "reduce_add"),
			&program,
			&context,
			&queue,
			&device, "SSD")
				/ temps.size();

		float std_dev = sqrt(variance);

		//Sort values into buckets
		std::vector<int> buckets = bucket_sort(temps, minimum, maximum, &program, &context, &queue, &device, "Bucket");
		
		//Find percentiles and median
		bool mid = false;
		bool q1 = false;
		bool q3 = false;
		int middle = temps.size() / 2;
		int quart1 = temps.size() / 4;
		int quart3 = (temps.size() * 3) / 4;
 		int count = 0;

		printf("Average: %f\n", mean);
		printf("Minimum: %f\n", minimum);
		printf("Maximum: %f\n", maximum);
		printf("Standard deviation: %f\n", std_dev);

		//Iterate over each bucket until the number of values up to this bucket are the percentiles
		for (int i = 0; i < buckets.size(); i++) {
			count += buckets[i];
			if (!q1 && (count >= quart1)){
				q1 = true;
				printf("25th Percentile: %.1f\n", ((float)i / 10.0f) + minimum);
			}
			if (!mid && (count >= middle)){
				mid = true;
				printf("Median: %.1f\n", ((float)i / 10.0f) + minimum);
			}
			
			if (!q3 && (count >= quart3)) {
				q3 = true;
				printf("75th Percentile: %.1f\n", ((float)i / 10.0f) + minimum);
			}
		}
		
		//Show program execution time
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
		printf("Execution duration: %d\n", duration);
		getchar();
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
		getchar();
	}

	return 0;
}
