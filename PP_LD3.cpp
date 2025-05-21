#define CL_HPP_TARGET_OPENCL_VERSION 300
// g++ -std=c++11 -I./include -L./lib -lOpenCL -o LD2 PP_LD2.cpp
// qsub -l nodes=1:ppn=1:gpus=1 -I

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#ifdef __linux__ 
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif
#pragma pack(1)

using namespace std;
using namespace cl;

CommandQueue queue;
Kernel kernel;
Context context;
Program program;
cl_float4* cpu_output;
Buffer cl_output;

// CONFIGURATION
#define IMG_WIDTH 1920
#define IMG_HEIGHT 1080
#define LOCAL_WORK_SIZE 64

struct Shape {
	cl_float3 scale;
	cl_float3 pos;
	cl_float3 color;
	int shape_type = 0; // 0 = sphere, 1 = plane, 2 = cilinder
	float diffuce = 0;
	float sheen = 0;
	float ambiant = 1.0f;
};

struct Light {
	cl_float3 color;
	cl_float3 dir;
};

void setup_opencl() {
	std::cout << "Setting up OpenCL...\n";
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	auto platform = platforms[0];
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	auto device = devices[0];
	std::ifstream renderer("renderer.cl");
	std::string src(std::istreambuf_iterator<char>(renderer), (std::istreambuf_iterator<char>()));
	Program::Sources sources(1, src.c_str());
	context = Context(devices);
	program = Program(context, sources);
	queue = CommandQueue(context, device);
	auto err = program.build();
	kernel = Kernel(program, "render_kernel");

	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		clGetProgramBuildInfo(program(), devices[0](), CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char* log = (char*)malloc(log_size);
		clGetProgramBuildInfo(program(), devices[0](), CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
	}
}

void render_image(
	cl_float3 sky_color, Light light, Shape* shapes, int shape_count,
	float shadow_strength = 1.5f, float ambiant_strength = 1.0f, float sheen_strength = 1.0f, float diffuce_strength = 1.0f, float base_color_strength = 0.0f
) {
	std::cout << "Rendering...\n";
	cpu_output = new cl_float3[IMG_WIDTH * IMG_HEIGHT];
	cl_output = Buffer(context, CL_MEM_WRITE_ONLY, IMG_WIDTH * IMG_HEIGHT * sizeof(cl_float3));
	Buffer shape_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Shape) * shape_count, shapes);
	std::vector<float> params = { shadow_strength, ambiant_strength, sheen_strength, diffuce_strength, base_color_strength };
	cl::Buffer paramBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * params.size(), params.data());
	kernel.setArg(0, cl_output);
	kernel.setArg(1, IMG_WIDTH);
	kernel.setArg(2, IMG_HEIGHT);
	kernel.setArg(3, sky_color);
	kernel.setArg(4, shape_buffer);
	kernel.setArg(5, shape_count);
	kernel.setArg(6, light);
	kernel.setArg(7, paramBuffer);
	queue.enqueueNDRangeKernel(kernel, NullRange, IMG_WIDTH * IMG_HEIGHT, LOCAL_WORK_SIZE);
	queue.enqueueReadBuffer(cl_output, CL_TRUE, 0, IMG_WIDTH * IMG_HEIGHT * sizeof(cl_float3), cpu_output);
	queue.finish();
}

inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
inline int toInt(float x) { return int(clamp(x) * 255 + .5); }
void saveImage(string name) { // save image to file .PPM
	std::cout << "Saving image...\n";
	name = name + ".ppm";
	FILE* f = fopen(name.c_str(), "w");
	if (f == 0) {
		perror("Failed to open file");
		return;
	}
	fprintf(f, "P3\n%d %d\n%d\n", IMG_WIDTH, IMG_HEIGHT, 255);

	for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
		fprintf(f, "%d %d %d ", toInt(cpu_output[i].s[0]), toInt(cpu_output[i].s[1]), toInt(cpu_output[i].s[2]));
	}
}

cl_float3 make_float3(float x = 0, float y = 0, float z = 0) {
	cl_float3 f;
	f.s[0] = x;
	f.s[1] = y;
	f.s[2] = z;
	f.s[3] = 0.0f;
	return f;
}

Light makeTheSUN(cl_float3 color, cl_float3 dir) {
	float mag = sqrt(dir.s[0] * dir.s[0] + dir.s[1] * dir.s[1] + dir.s[2] * dir.s[2]);
	Light light;
	light.color = color;
	light.dir = make_float3(dir.s[0] / mag, dir.s[1] / mag, dir.s[2] / mag); // normalize direction
	return light;
}

int main() {
#if CL_DEVICE_IMAGE_SUPPORT == CL_FALSE 
	std::cout << "Images not supported on this OpenCL version.\n";
	return 0;
#endif

	std::cout << "Starting...\n";

	long long elapsed = 0;
	std::chrono::system_clock::time_point start, end;

	setup_opencl();

	Light sun = makeTheSUN(make_float3(1.0f, 0.8f, 0.8f), make_float3(-1.0f, -1.0f, 0.75f));
	Shape* shapes = new Shape[5];

	shapes[0].scale = make_float3(0.5f);
	shapes[0].pos = make_float3(0.0f, 0.0f, 3.0f);
	shapes[0].color = make_float3(0.75f, 0.0f, 0.05f);
	shapes[0].shape_type = 0;
	shapes[0].diffuce = 0.7f;
	shapes[0].sheen = 0.1f;
	shapes[0].ambiant = 0.5f;
	shapes[1].scale = make_float3(0.2f);
	shapes[1].pos = make_float3(-0.5f, -0.2f, 2.5f);
	shapes[1].color = make_float3(0.1f, 0.1f, 0.9f);
	shapes[1].shape_type = 0;
	shapes[1].diffuce = 0.4f;
	shapes[1].sheen = 0.4f;
	shapes[1].ambiant = 0.7f;
	shapes[2].scale = make_float3(0.2f);
	shapes[2].pos = make_float3(0.5f, 0.3f, 3.5f);
	shapes[2].color = make_float3(0.1f, 0.1f, 0.85f);
	shapes[2].diffuce = 0.4f;
	shapes[2].sheen = 0.4f;
	shapes[2].ambiant = 0.7f;
	shapes[3].scale = make_float3(0.5f);
	shapes[3].pos = make_float3(1.5f, 0.0f, 4.0f);
	shapes[3].color = make_float3(0.95f, 0.95f, 0.95f);
	shapes[3].shape_type = 2;
	shapes[3].diffuce = 0.7f;
	shapes[3].sheen = 0.1f;
	shapes[3].ambiant = 0.8f;
	shapes[4].scale = make_float3(5.0f, 5.0f);
	shapes[4].pos = make_float3(0.0f, -1.2f, 3.0f);
	shapes[4].color = make_float3(0.1f, 0.5f, 0.1f);
	shapes[4].shape_type = 1; 
	shapes[4].diffuce = 0.5f;
	shapes[4].sheen = 0.05f;
	shapes[4].ambiant = 0.8f;

	start = std::chrono::system_clock::now();
	render_image(make_float3(0.7f, 0.7f, 0.9f), sun, shapes, 5, 0, 0, 0, 0, 1);
	std::cout << "Scene Rendered!\n";
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Elapsed time = " << elapsed << "ms\n";
	saveImage("res1");

	start = std::chrono::system_clock::now();
	render_image(make_float3(0.7f, 0.7f, 0.9f), sun, shapes, 5, 0);
	std::cout << "Scene Rendered!\n";
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Elapsed time = " << elapsed << "ms\n";
	saveImage("res2");

	start = std::chrono::system_clock::now();
	render_image(make_float3(0.7f, 0.7f, 0.9f), sun, shapes, 5);
	std::cout << "Scene Rendered!\n";
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Elapsed time = " << elapsed << "ms\n";
	saveImage("res3");

	delete[] cpu_output;
	std::cout << "Done!\n";
}

