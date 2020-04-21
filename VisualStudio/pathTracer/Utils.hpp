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
#include <glm/vec3.hpp>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using glm::vec3;

const float inf = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

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

	__host__ __device__ gvec3& operator+=(const gvec3 &v2);
	__host__ __device__ gvec3& operator-=(const gvec3 &v2);
	__host__ __device__ gvec3& operator*=(const gvec3 &v2);
	__host__ __device__ gvec3& operator/=(const gvec3 &v2);
	__host__ __device__ gvec3& operator*=(const float t);
	__host__ __device__ gvec3& operator/=(const float t);
};

inline std::istream& operator>>(std::istream &is, gvec3 &t) {
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

inline std::ostream& operator<<(std::ostream &os, const gvec3 &t) {
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

inline gvec3 operator+(const gvec3 &v1, const gvec3 &v2) {
	return gvec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline gvec3 operator-(const gvec3 &v1, const gvec3 &v2) {
	return gvec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline gvec3 operator*(const gvec3 &v1, const gvec3 &v2) {
	return gvec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

inline gvec3 operator/(const gvec3 &v1, const gvec3 &v2) {
	return gvec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

inline gvec3 operator*(float t, const gvec3 &v) {
	return gvec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline gvec3 operator/(gvec3 v, float t) {
	return gvec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

inline gvec3 operator*(const gvec3 &v, float t) {
	return gvec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline float dot(const gvec3 &v1, const gvec3 &v2) {
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

inline gvec3 cross(const gvec3 &v1, const gvec3 &v2) {
	return gvec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		(-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
		(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}


inline gvec3& gvec3::operator+=(const gvec3 &v) {
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

inline gvec3& gvec3::operator*=(const gvec3 &v) {
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

inline gvec3& gvec3::operator/=(const gvec3 &v) {
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

inline gvec3& gvec3::operator-=(const gvec3& v) {
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

inline gvec3& gvec3::operator*=(const float t) {
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

inline gvec3& gvec3::operator/=(const float t) {
	float k = 1.0 / t;

	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}

__host__ __device__ inline float length2(const gvec3& v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ inline float length(const gvec3& v) {
	return sqrt(length2(v));
}

__host__ __device__ inline gvec3 normalize(const gvec3& v) {
	return v / length(v);
}

__host__ __device__ float urand(float min, float max);
__host__ __device__ gvec3 reflect(const gvec3& i, const gvec3& n);
__host__ __device__ gvec3 refract(const gvec3& i, const gvec3& n, float k1, float k2);
__host__ __device__ float reflectance(const gvec3& i, const gvec3& n, float k1, float k2);

struct Profiler {
  std::chrono::high_resolution_clock::time_point t0;
  std::string name;
  
  Profiler(std::string const& n)
  : name(n), t0(std::chrono::high_resolution_clock::now()) { }
  
  ~Profiler() {
    auto diff = std::chrono::high_resolution_clock::now() - t0;
    std::cout << "\n" << name << ": "
    << std::chrono::duration_cast<std::chrono::minutes>(diff).count()
    << " min."
    << std::endl;
  }
};

#endif /* Utils_hpp */
