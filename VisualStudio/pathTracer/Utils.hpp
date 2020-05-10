//
//  Utils.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Utils_hpp
#define Utils_hpp

#include <iostream>
#include <chrono>
#include <string>
#include "cuda_runtime.h"
#include "curand_kernel.h"

__constant__ const float pi = 3.1415926535897932385f;
__constant__ const float inf = std::numeric_limits<float>::infinity();


#define CHK_CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__ )
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "\nCUDA error = " << (unsigned int)result << " at " <<
			file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(-1);
	}
}

class gvec3 {
public:
	union {
		float e[3];
		struct {
			union { float x, r, s; };
			union { float y, g, t; };
			union { float z, b, p; };
		};
	};

	__host__ __device__ gvec3(): x(0.f), y(0.f), z(0.f) {}
	__host__ __device__ gvec3(float x) : x(x), y(x), z(x) {}
	__host__ __device__ gvec3(float x, float y, float z) : x(x), y(y), z(z) {}
	__host__ __device__ const gvec3& operator+() const { return *this; }
	__host__ __device__ gvec3 operator-() const { return gvec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ float operator[](int i) const { return e[i]; }
	__host__ __device__ float& operator[](int i) { return e[i]; };
	__host__ __device__ gvec3& operator+=(const gvec3 &v);
	__host__ __device__ gvec3& operator*=(const gvec3 &v);
	__host__ __device__ gvec3& operator/=(const gvec3 &v);
	__host__ __device__ gvec3& operator-=(const gvec3& v);
	__host__ __device__ gvec3& operator*=(const float t);
	__host__ __device__ gvec3& operator/=(const float t);
};

std::istream& operator>>(std::istream &is, gvec3 &t);
std::ostream& operator<<(std::ostream &os, const gvec3 &t);

__host__ __device__ gvec3 operator+(const gvec3 &v1, const gvec3 &v2);
__host__ __device__ gvec3 operator-(const gvec3 &v1, const gvec3 &v2);
__host__ __device__ gvec3 operator*(const gvec3 &v1, const gvec3 &v2);
__host__ __device__ gvec3 operator/(const gvec3 &v1, const gvec3 &v2);
__host__ __device__ gvec3 operator*(float t, const gvec3 &v);
__host__ __device__ gvec3 operator/(gvec3 v, float t);
__host__ __device__ gvec3 operator*(const gvec3 &v, float t);
__host__ __device__ float dot(const gvec3 &v1, const gvec3 &v2);
__host__ __device__ gvec3 cross(const gvec3 &v1, const gvec3 &v2);
__host__ __device__ float length2(const gvec3& v);
__host__ __device__ float length(const gvec3& v);
__host__ __device__ gvec3 normalize(const gvec3& v);

__device__ gvec3 diskRand(float radius, curandState* rs);
__device__ gvec3 ballRand(float radius, curandState* rs);
__device__ gvec3 sphericalRand(float radius, curandState* rs);
__device__ gvec3 urand3(curandState* rs);

__device__ gvec3 reflect(const gvec3& i, const gvec3& n);
__device__ gvec3 refract(const gvec3& i, const gvec3& n, float k1, float k2);
__device__ float reflectance(const gvec3& i, const gvec3& n, float k1, float k2);

struct Profiler {
  std::chrono::high_resolution_clock::time_point t0;
  std::string name;
  
  Profiler(std::string const& n)
  : name(n), t0(std::chrono::high_resolution_clock::now()) { }
  
  ~Profiler() {
    auto diff = std::chrono::high_resolution_clock::now() - t0;
    std::cout << "\n" << name << ": "
    << std::chrono::duration_cast<std::chrono::seconds>(diff).count()
    << " sec."
    << std::endl;
  }
};

#endif /* Utils_hpp */
