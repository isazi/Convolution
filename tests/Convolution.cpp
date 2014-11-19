// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ctime>

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <Convolution.hpp>

typedef float dataType;
std::string typeName("float");


int main(int argc, char *argv[]) {
  bool print = false;
  bool random = false;
  bool localMem = false;
  unsigned int padding = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int nrColumnsPerBlock = 0;
	unsigned int nrRowsPerBlock = 0;
  unsigned int nrColumnsPerThread = 0;
  unsigned int nrRowsPerThread = 0;
  long long unsigned int wrongItems = 0;
  // I/O size
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int filterWidth = 0;
  unsigned int filterHeight = 0;

  try {
    isa::utils::ArgumentList args(argc, argv);
    print = args.getSwitch("-print");
    random = args.getSwitch("-random");
    localMem = args.getSwitch("-local");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
		nrColumnsPerBlock = args.getSwitchArgument< unsigned int >("-cb");
		nrRowsPerBlock = args.getSwitchArgument< unsigned int >("-rb");
		nrColumnsPerThread = args.getSwitchArgument< unsigned int >("-ct");
		nrRowsPerThread = args.getSwitchArgument< unsigned int >("-rt");
    width = args.getSwitchArgument< unsigned int >("-width");
    height = args.getSwitchArgument< unsigned int >("-height");
    filterWidth = args.getSwitchArgument< unsigned int >("-filter_width");
    filterHeight = args.getSwitchArgument< unsigned int >("-filter_height");

	} catch  ( isa::utils::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-print] [-random] [-local] -opencl_platform ... -opencl_device ... -padding ... -cb ... -rb ... -ct ... -rt ... -width ... -height ... -filter_width ... -filter_height ..." << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate host memory
  std::vector< dataType > input = std::vector< dataType >((width + (filterWidth - 1)) * (height + (filterHeight - 1)));
  std::vector< dataType > output = std::vector< dataType >(width * height);
  std::vector< dataType > output_c = std::vector< dataType >(width * height);
  std::vector< dataType > filter = std::vector< dataType >(filterWidth * filterHeight);
  if ( random ) {
    std::srand(time(0));
  } else {
    std::srand(42);
  }
  std::fill(filter.begin(), filter.end(), std::rand() % 100);
  std::fill(input.begin(), input.end(), std::rand() % 1000);

  // Allocate device memory
  cl::Buffer input_d, output_d, filter_d;
  try {
    input_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, input.size() * sizeof(dataType), 0, 0);
    output_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, output.size() * sizeof(dataType), 0, 0);
    filter_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, filter.size() * sizeof(dataType), 0, 0);
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString(err.err()) << "." << std::endl;
    return 1;
  }

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(filter_d, CL_FALSE, 0, filter.size() * sizeof(float), reinterpret_cast< void * >(filter.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input.size() * sizeof(dataType), reinterpret_cast< void * >(input.data()));
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString(err.err()) << "." << std::endl;
    return 1;
  }

	// Generate kernel
  std::string * code = isa::OpenCL::getConvolutionOpenCL(localMem, padding, width, height, filterWidth, filterHeight, nrColumnsPerBlock, nrRowsPerBlock, nrColumnsPerThread, nrRowsPerThread, typeName);
  cl::Kernel * kernel;
  if ( print ) {
    std::cout << *code << std::endl;
  }
	try {
    kernel = isa::OpenCL::compile("convolution", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
	} catch ( isa::OpenCL::OpenCLError &err ) {
    std::cerr << err.what() << std::endl;
		return 1;
	}

  // Run OpenCL kernel and CPU control
  try {
    cl::NDRange global(width / nrColumnsPerThread, height / nrRowsPerThread);
    cl::NDRange local(nrColumnsPerBlock, nrRowsPerBlock);

    kernel->setArg(0, input_d);
    kernel->setArg(1, output_d);
    kernel->setArg(2, filter_d);
    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
    isa::OpenCL::convolution< dataType >(padding, width, height, filterWidth, filterHeight, input, output_c, filter);
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(dataType), reinterpret_cast< void * >(output.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error kernel execution: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  for ( unsigned int y = 0; y < height; y++ ) {
    for ( unsigned int x = 0; x < width; x++ ) {
      if ( !isa::utils::same(output[(y * isa::utils::pad(width, padding)) + x], output_c[(y * isa::utils::pad(width, padding)) + x]) ) {
        wrongItems++;
      }
    }
  }

  if ( wrongItems > 0 ) {
    std::cout << "Wrong input: " << wrongItems << " (" << (wrongItems * 100.0) / (static_cast< long long unsigned int >(width) * height) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

