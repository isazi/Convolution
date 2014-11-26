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
#include <algorithm>

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <Convolution.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>

typedef float dataType;
std::string typeName("float");


int main(int argc, char * argv[]) {
  bool localMem = false;
	unsigned int nrIterations = 0;
  unsigned int padding = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int minThreads = 0;
  unsigned int maxThreads = 0;
	unsigned int maxRows = 0;
	unsigned int maxColumns = 0;
  unsigned int threadUnit = 0;
  unsigned int threadIncrement = 0;
  unsigned int maxItems = 0;
  // I/O size
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int filterWidth = 0;
  unsigned int filterHeight = 0;

	try {
    isa::utils::ArgumentList args(argc, argv);

    localMem = args.getSwitch("-local");
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    threadUnit = args.getSwitchArgument< unsigned int >("-thread_unit");
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
		maxRows = args.getSwitchArgument< unsigned int >("-max_rows");
		maxColumns = args.getSwitchArgument< unsigned int >("-max_columns");
    threadIncrement = args.getSwitchArgument< unsigned int >("-thread_increment");
		maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    width = args.getSwitchArgument< unsigned int >("-width");
    height = args.getSwitchArgument< unsigned int >("-height");
    filterWidth = args.getSwitchArgument< unsigned int >("-filter_width");
    filterHeight = args.getSwitchArgument< unsigned int >("-filter_height");
	} catch ( isa::utils::EmptyCommandLine & err ) {
		std::cerr << argv[0] << " -iterations ... [-local] -opencl_platform ... -opencl_device ... -padding ... -thread_unit ... -min_threads ... -max_threads ... -max_items ... -max_columns ... -max_rows ... -thread_increment ... -width ... -height ... -filter_width ... -filter_height ..." << std::endl;
		return 1;
	} catch ( std::exception & err ) {
		std::cerr << err.what() << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate host memory
  std::vector< dataType > input = std::vector< dataType >((height + (filterHeight - 1)) * isa::utils::pad(width + (filterWidth - 1), padding));
  std::vector< dataType > output = std::vector< dataType >(height * isa::utils::pad(width, padding));
  std::vector< dataType > filter = std::vector< dataType >(filterWidth * filterHeight);
  std::srand(time(0));
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

	// Find the parameters
	std::vector< unsigned int > columnsPerBlock;
	for ( unsigned int columns = minThreads; columns <= maxColumns; columns += threadIncrement ) {
		if ( (width % columns) == 0 ) {
			columnsPerBlock.push_back(columns);
		}
	}
	std::vector< unsigned int > rowsPerBlock;
	for ( unsigned int rows = 1; rows <= maxRows; rows++ ) {
		if ( (height % rows) == 0 ) {
			rowsPerBlock.push_back(rows);
		}
	}

	std::cout << std::fixed << std::endl;
	std::cout << "# width height filterWidth filterHeight local columnsPerBlock rowsPerBlock columnsPerThread rowsPerThread GFLOP/s GB/s time stdDeviation COV" << std::endl << std::endl;

	for ( std::vector< unsigned int >::iterator columns = columnsPerBlock.begin(); columns != columnsPerBlock.end(); ++columns ) {
		for ( std::vector< unsigned int >::iterator rows = rowsPerBlock.begin(); rows != rowsPerBlock.end(); ++rows ) {
			if ( ((*columns) * (*rows)) > maxThreads ) {
				break;
			} else if ( ((*columns) * (*rows)) % threadUnit != 0 ) {
        continue;
      }

			for ( unsigned int columnsPerThread = 1; columnsPerThread <= maxItems; columnsPerThread++ ) {
				if ( (width % ((*columns) * columnsPerThread)) != 0 ) {
					continue;
				}

				for ( unsigned int rowsPerThread = 1; rowsPerThread <= maxItems; rowsPerThread++ ) {
					if ( (height % ((*rows) * rowsPerThread)) != 0 ) {
						continue;
					} else if ( !localMem && (columnsPerThread * rowsPerThread) + 2 > maxItems ) {
						break;
					} else if ( localMem  && (columnsPerThread * rowsPerThread) + 5 > maxItems ) {
            break;
          }

          // Generate kernel
          double gflops = isa::utils::giga((static_cast< long long unsigned int >(width) * height * filterWidth * filterHeight * 2) + (static_cast< long long unsigned int >(width) * height));
          double gbs;
          if ( localMem ) {
            gbs = isa::utils::giga((static_cast< long long unsigned int >(width / (*columns * columnsPerThread)) * (height / (*rows * rowsPerThread)) * (((*columns * columnsPerThread) + (filterWidth - 1)) * ((*rows * rowsPerThread) + (filterHeight - 1))) * sizeof(dataType)) + (static_cast< long long unsigned int >(width) * height * sizeof(dataType)) + (static_cast< long long unsigned int >(width) * (height) * filterWidth * filterHeight * sizeof(dataType)));
          } else {
            gbs = isa::utils::giga((static_cast< long long unsigned int >(width) * height * filterWidth * filterHeight * 2 * sizeof(dataType)) + (static_cast< long long unsigned int >(width) * height * sizeof(dataType)));
          }
          isa::utils::Timer timer;
          cl::Event event;
          cl::Kernel * kernel;
          std::string * code = isa::OpenCL::getConvolutionOpenCL(localMem, padding, width, height, filterWidth, filterHeight, *columns, *rows, columnsPerThread, rowsPerThread, typeName);

          try {
            kernel = isa::OpenCL::compile("convolution", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
          } catch ( isa::OpenCL::OpenCLError & err ) {
            std::cerr << *columns << ", " << *rows << ", " << columnsPerThread << ", " << rowsPerThread << std::endl;
            std::cerr << err.what() << std::endl;
            continue;
          }

          cl::NDRange global(width / columnsPerThread, height / rowsPerThread);
          cl::NDRange local(*columns, *rows);

          kernel->setArg(0, input_d);
          kernel->setArg(1, output_d);
          kernel->setArg(2, filter_d);

          // Warm-up run
          try {
            clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
            event.wait();
          } catch ( cl::Error & err ) {
            std::cerr << *columns << ", " << *rows << ", " << columnsPerThread << ", " << rowsPerThread << std::endl;
            std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
            continue;
          }
          // Tuning runs
          try {
            for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
              timer.start();
              clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
              event.wait();
              timer.stop();
            }
          } catch ( cl::Error & err ) {
            std::cerr << *columns << ", " << *rows << ", " << columnsPerThread << ", " << rowsPerThread << std::endl;
            std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
            continue;
          }

          std::cout << width << " " << height << " " << filterWidth << " " << filterHeight << " ";
          std::cout << localMem << " " << *columns << " " << *rows << " " << columnsPerThread << " " << rowsPerThread << " ";
          std::cout << std::setprecision(3);
          std::cout << gflops / timer.getAverageTime() << " ";
          std::cout << gbs / timer.getAverageTime() << " ";
          std::cout << std::setprecision(6);
          std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " ";
          std::cout << timer.getCoefficientOfVariation() <<  std::endl;
				}
			}
		}
	}

	std::cout << std::endl;

	return 0;
}

