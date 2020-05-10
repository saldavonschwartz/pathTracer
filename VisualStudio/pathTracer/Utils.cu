//
//  Utils.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Utils.hpp"

__host__ __device__ gvec3& gvec3::operator+=(const gvec3 &v) {
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

__host__ __device__ gvec3& gvec3::operator*=(const gvec3 &v) {
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

__host__ __device__ gvec3& gvec3::operator/=(const gvec3 &v) {
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

__host__ __device__ gvec3& gvec3::operator-=(const gvec3& v) {
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

__host__ __device__ gvec3& gvec3::operator*=(const float t) {
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ gvec3& gvec3::operator/=(const float t) {
	float k = 1.f / t;

	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}

std::istream& operator>>(std::istream &is, gvec3 &t) {
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

std::ostream& operator<<(std::ostream &os, const gvec3 &t) {
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

__host__ __device__ gvec3 operator+(const gvec3 &v1, const gvec3 &v2) {
	return gvec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ gvec3 operator-(const gvec3 &v1, const gvec3 &v2) {
	return gvec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ gvec3 operator*(const gvec3 &v1, const gvec3 &v2) {
	return gvec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ gvec3 operator/(const gvec3 &v1, const gvec3 &v2) {
	return gvec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ gvec3 operator*(float t, const gvec3 &v) {
	return gvec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ gvec3 operator/(gvec3 v, float t) {
	return gvec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ gvec3 operator*(const gvec3 &v, float t) {
	return gvec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ float dot(const gvec3 &v1, const gvec3 &v2) {
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ gvec3 cross(const gvec3 &v1, const gvec3 &v2) {
	return gvec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		(-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
		(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ float length2(const gvec3& v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ float length(const gvec3& v) {
	return sqrt(length2(v));
}

__host__ __device__ gvec3 normalize(const gvec3& v) {
	return v / length(v);
}

__device__ gvec3 diskRand(float radius, curandState* rs) {
	gvec3 p;

	do {
		p = gvec3{
			curand_uniform(rs) * 2.f - 1.f,
			curand_uniform(rs) * 2.f - 1.f,
			0.f
		};

	} while (dot(p, p) >= 1);

	return p * radius;
}

__device__ gvec3 ballRand(float radius, curandState* rs) {
	gvec3 p;

	do {
		p = gvec3{
			curand_uniform(rs) * 2.f - 1.f,
			curand_uniform(rs) * 2.f - 1.f,
			curand_uniform(rs) * 2.f - 1.f
		};

	} while (dot(p, p) >= 1.f);

	return p * radius;
}

__device__ gvec3 sphericalRand(float radius, curandState* rs) {
	float a = curand_uniform(rs) * 2.f * pi;
	float z = curand_uniform(rs) * 2.f - 1.f;
	float r = sqrtf(1.f - float(z * z));
	return gvec3(r*cosf(a), r*sinf(a), z) * radius;
}

__device__ gvec3 urand3(curandState* rs) {
	return {
		curand_uniform(rs),
		curand_uniform(rs),
		curand_uniform(rs)
	};
}

__device__ gvec3 reflect(const gvec3& i, const gvec3& n) {
	return i + 2.f * -dot(i, n) * n;
}

// Source: https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
__device__ gvec3 refract(const gvec3& i, const gvec3& n, float k1, float k2) {
	auto k = k1 / k2;
	auto cosi = -dot(i, n);
	auto sin2t = k * k * (1.f - cosi * cosi);
	return k * i + (k*cosi - sqrt(1.f - sin2t))*n;
}

// Source: https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
__device__ float reflectance(const gvec3& i, const gvec3& n, float k1, float k2) {
	auto r0 = (k1 - k2) / (k1 + k2);
	r0 *= r0;
	auto cosi = -dot(i, n);

	// Inside medium with higher refractive index:
	if (k1 > k2) {
		auto k = k1 / k2;
		auto sin2t = k * k * (1.f - cosi * cosi);

		// And total internal reflectance (TIR):
		if (sin2t > 1) {
			return 1;
		}
	}

	// Inside medium with lower refractive index or
	// Inside medium with higher refractive index but below critical incidence angle (no TIR):
	auto x = (1.f - cosi);
	return r0 + (1.f - r0) * x * x * x * x * x;
}
