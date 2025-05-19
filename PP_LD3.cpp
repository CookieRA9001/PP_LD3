#define CL_HPP_TARGET_OPENCL_VERSION 300
// g++ -std=c++11 -I./include -L./lib -lOpenCL -o LD2 PP_LD2.cpp
// qsub - l nodes = 1:ppn = 1 : gpus = 1 - I

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#ifdef __linux__ 
	#include <CL/cl.hpp>
#else
	#include <CL/opencl.hpp>
#endif

cl::Program setup() {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	auto platform = platforms[0];
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	auto device = devices[0];

	std::ifstream matecarlo("matecarlo.cl");
	std::string src(std::istreambuf_iterator<char>(matecarlo), (std::istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, src.c_str());
	cl::Context context(devices);
	cl::Program program(context, sources);
	auto err = program.build("-cl-std=CL3.0");

	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		clGetProgramBuildInfo(program(), devices[0](), CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char* log = (char*)malloc(log_size);
		clGetProgramBuildInfo(program(), devices[0](), CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
	}

	return program;
}

float func(float x, int a, int b, int c) { return (x - a) * (x - b) * (x - c); }

float montecarlo(
	int xmin, int xmax,
	int ymin, int ymax,
	int a, int b, int c,
	const size_t points
) {
	signed long long int count = 0;
	float x, y, f;
	for (int i = 0; i < points; i++) {
		y = ymin + static_cast<float>(rand()) / RAND_MAX * (ymax - ymin);
		x = xmin + static_cast<float>(rand()) / RAND_MAX * (xmax - xmin);
		f = func(x, a, b, c);
		if (f > 0 && y < f && y > 0) count++;
		if (f < 0 && y > f && y < 0) count--;
	}
	return (float)count / (float)points * (xmax - xmin) * (ymax - ymin);
}

float opencl_montecarlo(
	cl::Program program, cl::Context context, cl::Device device,
	int xmin, int xmax,
	int ymin, int ymax,
	int a, int b, int c,
	const int points
) {
	int power_per_unit = std::max(points / CL_DEVICE_MAX_COMPUTE_UNITS,1);
	std::vector<int> buff_out(CL_DEVICE_MAX_COMPUTE_UNITS);
	std::vector<int> params = { a, b, c, xmin, xmax, ymin, ymax, power_per_unit };
	cl::Buffer outBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * buff_out.size(), nullptr);
	cl::Buffer paramBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * params.size(), params.data()); 
	cl::Kernel kernel(program, "matecarlo_fun");
	kernel.setArg(0, outBuffer);
	kernel.setArg(1, paramBuffer); 

	auto name = device.getInfo<CL_DEVICE_NAME>();
	std::cout << "Device: " << name << " " << CL_DEVICE_MAX_COMPUTE_UNITS << "\n";

	cl::CommandQueue queue(context, device);  
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(buff_out.size()));
	queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, sizeof(float) * buff_out.size(), buff_out.data());

	queue.finish();

	float count = 0.0f;
	for (int v : buff_out) count += v;
	return (float)count / (float)(points) * (xmax - xmin) * (ymax - ymin);;
}

int main() {
	#if CL_DEVICE_IMAGE_SUPPORT == CL_FALSE 
		std::cout << "Images not supported on this OpenCL version.\n";
		return 0;
	#endif

	std::cout << "Starting...\n";

	long long elapsed = 0;
	std::chrono::system_clock::time_point start, end;

	cl::Program program = setup();
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	cl::Device device = program.getInfo<CL_PROGRAM_DEVICES>()[0];

	/*std::cout << "\nParalel code with 1000 points:\n";
	start = std::chrono::system_clock::now();
	res = opencl_montecarlo( program, context, device, x_min, x_max, y_min, y_max, a, b, c, 1000);
	printf("Definite integral = %.5f\n", res);
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Elapsed time = " << elapsed << "ms\n";*/

	std::cin.get();
}
