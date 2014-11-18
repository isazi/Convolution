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

#include <string>

#include <utils.hpp>


#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

namespace isa {
namespace OpenCL {

// Sequential convolution algorithm
template< typename T > void convolution(const unsigned int padding, const unsigned int width, const unsigned int height, const unsigned int filterWidth, const unsigned int filterHeight, const std::vector< T > & input, std::vector< T > & output, const std::vector< T > & filter);

// Implementations
template< typename T > void convolution(const unsigned int padding, const unsigned int width, const unsigned int height, const unsigned int filterWidth, const unsigned int filterHeight, const std::vector< T > & input, std::vector< T > & output, const std::vector< T > & filter) {
  const unsigned int verticalBorder = filterHeight / 2;
  const unsigned int horizontalBorder = filterWidth / 2;

  for ( unsigned int x = 0; x < width; x++ ) {
    for ( unsigned int y = 0; y < height; y++ ) {
      T sum = 0;

      for (unsigned int fX = x - horizontalBorder; fX < x + horizontalBorder; fX++ ) {
        for ( unsigned int fY = y - verticalBorder; fY < y + verticalBorder; fY++ ) {
          sum += input[((x + horizontalBorder) * isa::utils::pad(width + (2 * horizontalBorder), padding)) + (y + verticalBorder)] * filter[(fX * filterWidth) + fY];
        }
      }
      sum /= filterWidth * filterHeight;
      output[(x * isa::utils::pad(width, padding)) + y] = sum;
    }
  }
}

} // OpenCL
} // isa

#endif // CONVOLUTION_HPP

