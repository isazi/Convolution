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
#include <cmath>

#include <utils.hpp>


#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

namespace isa {
namespace OpenCL {

// Sequential convolution algorithm
template< typename T > void convolution(const unsigned int padding, const unsigned int width, const unsigned int height, const unsigned int filterWidth, const unsigned int filterHeight, const std::vector< T > & input, std::vector< T > & output, const std::vector< T > & filter);
// OpenCL convolution algorithm
std::string * getConvolutionOpenCL(const bool local, const unsigned int padding, const unsigned int width, const unsigned int height, const unsigned int filterWidth, const unsigned int filterHeight, const unsigned int nrColumnsPerBlock, const unsigned int nrRowsPerBlock, const unsigned int nrColumnsPerThread, const unsigned int nrRowsPerThread, std::string & dataType);

// Implementations
template< typename T > void convolution(const unsigned int padding, const unsigned int width, const unsigned int height, const unsigned int filterWidth, const unsigned int filterHeight, const std::vector< T > & input, std::vector< T > & output, const std::vector< T > & filter) {

  for ( unsigned int y = 0; y < height; y++ ) {
    for ( unsigned int x = 0; x < width; x++ ) {
      T sum = 0;

      for ( unsigned int fY = y; fY < y + filterHeight; fY++ ) {
        for (unsigned int fX = x; fX < x + filterWidth; fX++ ) {
          sum += input[(fY * isa::utils::pad(width + (filterWidth - 1), padding)) + fX] * filter[((fY - y) * filterWidth) + (fX - x)];
        }
      }
      sum /= filterWidth * filterHeight;
      output[(y * isa::utils::pad(width, padding)) + x] = sum;
    }
  }
}

std::string * getConvolutionOpenCL(const bool local, const unsigned int padding, const unsigned int width, const unsigned int height, const unsigned int filterWidth, const unsigned int filterHeight, const unsigned int nrColumnsPerBlock, const unsigned int nrRowsPerBlock, const unsigned int nrColumnsPerThread, const unsigned int nrRowsPerThread, std::string & dataType) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void convolution(__global const " + dataType + " * const restrict input, __global " + dataType + " * const restrict output, __constant const " + dataType + " * const restrict filter) {\n";
  if ( local ) {
    *code += "const unsigned int x = (get_group_id(0) * " + isa::utils::toString(nrColumnsPerBlock * nrColumnsPerThread) + ");\n"
      "const unsigned int y = (get_group_id(1) * " + isa::utils::toString(nrRowsPerBlock * nrRowsPerThread) + ");\n"
      "unsigned int fX = 0;\n"
      "unsigned int fY = 0;\n";
    if ( nrColumnsPerBlock < padding ) {
      *code += "__local " + dataType + " localInput[" + isa::utils::toString(isa::utils::pad((nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1), padding) * ((nrRowsPerBlock * nrRowsPerThread) + (filterHeight - 1))) + "];\n";
    } else {
      *code += "__local " + dataType + " localInput[" + isa::utils::toString(((nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1)) * ((nrRowsPerBlock * nrRowsPerThread) + (filterHeight - 1))) + "];\n";
    }
  } else {
    *code += "const unsigned int x = (get_group_id(0) * " + isa::utils::toString(nrColumnsPerBlock * nrColumnsPerThread) + ") + get_local_id(0);\n"
      "const unsigned int y = (get_group_id(1) * " + isa::utils::toString(nrRowsPerBlock * nrRowsPerThread) + ") + get_local_id(1);\n";
  }
  *code += "<%DEF_SUMS%>";
  if ( local ) {
    *code += "fY = get_local_id(1);\n"
      "fX = get_local_id(0);\n"
      "<%LOAD%>"
      "barrier(CLK_LOCAL_MEM_FENCE);\n"
      "fY = get_local_id(1);\n"
      "fX = get_local_id(0);\n"
      "<%SUMS%>";
  } else {
    *code += "for ( unsigned int fY = y; fY < y + " + isa::utils::toString(filterHeight) + "; fY++ ) {\n"
      "for ( unsigned int fX = x; fX < x + " + isa::utils::toString(filterWidth) + "; fX++ ) {\n"
      "<%SUMS%>"
      "}\n"
      "}\n";
  }
  *code += "<%AVERAGE%>"
    "<%STORE%>"
    "}\n";
  std::string defSumsTemplate = dataType + " sumX<%XNUM%>Y<%YNUM%> = 0;\n";
  std::string loadTemplate;
  std::string sumsTemplate;
  if ( local ) {
    if ( nrColumnsPerBlock < padding ) {
      loadTemplate = "localInput[((fY + <%YOFFSET%>) * " + isa::utils::toString(isa::utils::pad((nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1), padding)) + ") + (fX + <%XOFFSET%>)] = input[((y + fY + <%YOFFSET%>) * " + isa::utils::toString(isa::utils::pad(width + (filterWidth - 1), padding)) + ") + (x + fX + <%XOFFSET%>)];\n";
      sumsTemplate = "sumX<%XNUM%>Y<%YNUM%> += localInput[((fY + <%YOFFSET%>) * " + isa::utils::toString(isa::utils::pad((nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1), padding)) + ") + (fX + <%XOFFSET%>)] * filter[((fY - get_local_id(1)) * " + isa::utils::toString(filterWidth) + ") + (fX - get_local_id(0))];\n";
    } else {
      loadTemplate = "localInput[((fY + <%YOFFSET%>) * " + isa::utils::toString((nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1)) + ") + (fX + <%XOFFSET%>)] = input[((y + fY + <%YOFFSET%>) * " + isa::utils::toString(isa::utils::pad(width + (filterWidth - 1), padding)) + ") + (x + fX + <%XOFFSET%>)];\n";
      sumsTemplate = "sumX<%XNUM%>Y<%YNUM%> += localInput[((fY + <%YOFFSET%>) * " + isa::utils::toString((nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1)) + ") + (fX + <%XOFFSET%>)] * filter[((fY - get_local_id(1)) * " + isa::utils::toString(filterWidth) + ") + (fX - get_local_id(0))];\n";
    }
  } else {
    sumsTemplate =  "sumX<%XNUM%>Y<%YNUM%> += input[((fY + <%YOFFSET%>) * " + isa::utils::toString(isa::utils::pad(width + (filterWidth - 1), padding)) + ") + (fX + <%XOFFSET%>)] * filter[((fY - y) * " + isa::utils::toString(filterWidth) + ") + (fX - x)];\n";
  }
  std::string loadYIncTemplate = "fY += " + isa::utils::toString(nrRowsPerBlock * nrRowsPerThread) + ";\n";
  std::string loadXIncTemplate = "fX += " + isa::utils::toString(nrColumnsPerBlock * nrColumnsPerThread) + ";\n";
  std::string xResetTemplate = "fX = get_local_id(0);\n";
  std::string sumYIncTemplate = "fY++;\n";
  std::string sumXIncTemplate = "fX++;\n";
  std::string averageTemplate = "sumX<%XNUM%>Y<%YNUM%> *= " + isa::utils::toString(1.0f / (filterWidth * filterHeight)) + "f;\n";
  std::string storeTemplate;
  if ( local ) {
    storeTemplate = "output[((y + get_local_id(1) + <%YOFFSET%>) * " + isa::utils::toString(isa::utils::pad(width, padding)) + ") + (x + get_local_id(0) + <%XOFFSET%>)] = sumX<%XNUM%>Y<%YNUM%>;\n";
  } else {
    storeTemplate = "output[((y + <%YOFFSET%>) * " + isa::utils::toString(isa::utils::pad(width, padding)) + ") + (x + <%XOFFSET%>)] = sumX<%XNUM%>Y<%YNUM%>;\n";
  }
  // End kernel's template

  std::string * defSums_s = new std::string();
  std::string * load_s = new std::string();
  std::string * sums_s = new std::string();
  std::string * average_s = new std::string();
  std::string * store_s = new std::string();

  for ( unsigned int y = 0; y < nrRowsPerThread; y++ ) {
    std::string y_s = isa::utils::toString(y);
    std::string yOffset_s = isa::utils::toString(y * nrRowsPerBlock);

    for ( unsigned int x = 0; x < nrColumnsPerThread; x++ ) {
      std::string x_s = isa::utils::toString(x);
      std::string xOffset_s = isa::utils::toString(x * nrColumnsPerBlock);
      std::string * temp_s = 0;

      temp_s = isa::utils::replace(&defSumsTemplate, "<%XNUM%>", x_s);
      temp_s = isa::utils::replace(temp_s, "<%YNUM%>", y_s, true);
      defSums_s->append(*temp_s);
      delete temp_s;
      if ( !local ) {
        temp_s = isa::utils::replace(&sumsTemplate, "<%XNUM%>", x_s);
        temp_s = isa::utils::replace(temp_s, "<%YNUM%>", y_s, true);
        temp_s = isa::utils::replace(temp_s, "<%XOFFSET%>", xOffset_s, true);
        temp_s = isa::utils::replace(temp_s, "<%YOFFSET%>", yOffset_s, true);
        sums_s->append(*temp_s);
        delete temp_s;
      }
      temp_s = isa::utils::replace(&averageTemplate, "<%XNUM%>", x_s);
      temp_s = isa::utils::replace(temp_s, "<%YNUM%>", y_s, true);
      average_s->append(*temp_s);
      delete temp_s;
      temp_s = isa::utils::replace(&storeTemplate, "<%XNUM%>", x_s);
      temp_s = isa::utils::replace(temp_s, "<%YNUM%>", y_s, true);
      temp_s = isa::utils::replace(temp_s, "<%XOFFSET%>", xOffset_s, true);
      temp_s = isa::utils::replace(temp_s, "<%YOFFSET%>", yOffset_s, true);
      store_s->append(*temp_s);
      delete temp_s;
    }
  }
  
  for ( unsigned int j = 0; j < static_cast< unsigned int >(std::ceil(((nrRowsPerBlock * nrRowsPerThread) + (filterHeight - 1)) / static_cast< float >(nrRowsPerBlock * nrRowsPerThread))); j++ ) {
    const unsigned int rows = (nrRowsPerBlock * nrRowsPerThread) + (filterHeight - 1);

    for ( unsigned int i = 0; i < static_cast< unsigned int >(std::ceil(((nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1)) / static_cast< float >(nrColumnsPerBlock * nrColumnsPerThread))); i++ ) {
      const unsigned int columns = (nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1);

      for ( unsigned int y = 0; y < nrRowsPerThread; y++ ) {
        std::string yOffset_s = isa::utils::toString(y * nrRowsPerBlock);

        for ( unsigned int x = 0; x < nrColumnsPerThread; x++ ) {
          std::string xOffset_s = isa::utils::toString(x * nrColumnsPerBlock);
          std::string * temp_s = new std::string();

          if ( (j == static_cast< unsigned int >(std::ceil(((nrRowsPerBlock * nrRowsPerThread) + (filterHeight - 1)) / static_cast< float >(nrRowsPerBlock * nrRowsPerThread))) - 1) && (i == static_cast< unsigned int >(std::ceil(((nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1)) / static_cast< float >(nrColumnsPerBlock * nrColumnsPerThread))) - 1) ){
            temp_s->append("if ( (fY + <%YOFFSET%>) < " + isa::utils::toString(rows) + " && (fX + <%XOFFSET%>) < " + isa::utils::toString(columns) + " ) {\n" + loadTemplate + "}\n");
          } else if ( j == static_cast< unsigned int >(std::ceil(((nrRowsPerBlock * nrRowsPerThread) + (filterHeight - 1)) / static_cast< float >(nrRowsPerBlock * nrRowsPerThread))) - 1 ) {
            temp_s->append("if ( (fY + <%YOFFSET%>) < " + isa::utils::toString(rows) + " ) {\n" + loadTemplate + "}\n");
          } else if ( i == static_cast< unsigned int >(std::ceil(((nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1)) / static_cast< float >(nrColumnsPerBlock * nrColumnsPerThread))) - 1 ) {
            temp_s->append("if ( (fX + <%XOFFSET%>) < " + isa::utils::toString(columns) + " ) {\n" + loadTemplate + "}\n");
          } else {
            temp_s->append(loadTemplate);
          }
          temp_s = isa::utils::replace(temp_s, "<%XOFFSET%>", xOffset_s, true);
          temp_s = isa::utils::replace(temp_s, "<%YOFFSET%>", yOffset_s, true);
          load_s->append(*temp_s);
          delete temp_s;
        }
      }
      if ( i != static_cast< unsigned int >(std::ceil(((nrColumnsPerBlock * nrColumnsPerThread) + (filterWidth - 1)) / static_cast< float >(nrColumnsPerBlock * nrColumnsPerThread))) - 1 ) {
        load_s->append(loadXIncTemplate);
      }
    }
    if ( j != static_cast< unsigned int >(std::ceil(((nrRowsPerBlock * nrRowsPerThread) + (filterHeight - 1)) / static_cast< float >(nrRowsPerBlock * nrRowsPerThread))) - 1 ) {
      load_s->append(loadYIncTemplate);
      load_s->append(xResetTemplate);
    }
  }

  if ( local ) {
    for ( unsigned int j = 0; j < filterHeight; j++ ) {
      for ( unsigned int i = 0; i < filterWidth; i++ ) {
        for ( unsigned int y = 0; y < nrRowsPerThread; y++ ) {
          std::string y_s = isa::utils::toString(y);
          std::string yOffset_s = isa::utils::toString(y * nrRowsPerBlock);

          for ( unsigned int x = 0; x < nrColumnsPerThread; x++ ) {
            std::string x_s = isa::utils::toString(x);
            std::string xOffset_s = isa::utils::toString(x * nrColumnsPerBlock);
            std::string * temp_s = 0;

            temp_s = isa::utils::replace(&sumsTemplate, "<%XNUM%>", x_s);
            temp_s = isa::utils::replace(temp_s, "<%YNUM%>", y_s, true);
            temp_s = isa::utils::replace(temp_s, "<%XOFFSET%>", xOffset_s, true);
            temp_s = isa::utils::replace(temp_s, "<%YOFFSET%>", yOffset_s, true);
            sums_s->append(*temp_s);
            delete temp_s;
          }
        }
        if ( i != filterWidth - 1 ) {
          sums_s->append(sumXIncTemplate);
        }
      }
      if ( j != filterHeight - 1 ) {
        sums_s->append(sumYIncTemplate);
        sums_s->append(xResetTemplate);
      }
    }
  }

  code = isa::utils::replace(code, "<%DEF_SUMS%>", *defSums_s, true);
  code = isa::utils::replace(code, "<%LOAD%>", *load_s, true);
  code = isa::utils::replace(code, "<%SUMS%>", *sums_s, true);
  code = isa::utils::replace(code, "<%AVERAGE%>", *average_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete defSums_s;
  delete load_s;
  delete sums_s;
  delete average_s;
  delete store_s;

  return code;
}

} // OpenCL
} // isa

#endif // CONVOLUTION_HPP

