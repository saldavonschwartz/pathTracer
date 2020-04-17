//
//  Utils.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Utils.hpp"
#include <glm/glm.hpp>

using namespace std;
using namespace glm;

static uniform_real_distribution<double> uniform01(0., 1.);
static mt19937 generator;

double urand(double min, double max) {
  return min + (max-min)*uniform01(generator);
};

vec3 to01Range(const vec3& v, bool normalize) {
  vec3 v_ = normalize ? glm::normalize(v) : v;
  return 0.5f * (v_ + vec3{1});
}

// Using GLM versions for now...
//vec3 randDisk() {
//  vec3 p;
//
//  do {
//    p = vec3{randUniform(-1, 1), randUniform(-1, 1), 0};
//  } while (dot(p, p) >= 1);
//
//  return p;
//}
//
//vec3 randBall() {
//  vec3 p;
//
//  do {
//    p = vec3{randUniform(-1, 1), randUniform(-1, 1), randUniform(-1, 1)};
//  } while (dot(p, p) >= 1);
//
//  return p;
//}
//
//vec3 randHemisphere(const vec3& normal) {
//  vec3 v = randSphere();
//  return dot(v, normal) >= 0.f ? v : -v;
//}
//
//vec3 randSphere() {
//  auto a = randUniform(0, 2*pi);
//  auto z = randUniform(-1, 1);
//  auto r = sqrt(1 - z*z);
//  return vec3(r*cos(a), r*sin(a), z);
//}

vec3 reflect(const vec3& i, const vec3& n) {
  return i + 2.f * -dot(i, n) * n;
}

// Source: https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
vec3 refract(const vec3& i, const vec3& n, float k1, float k2) {
  auto k = k1/k2;
  auto cosi = -dot(i, n);
  auto sin2t = k*k * (1.f - cosi*cosi);
  assert(sin2t <= 1);
  return k*i + (k*cosi - sqrt(1.f-sin2t))*n;
}

// Source: https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
float reflectance(const vec3& i, const vec3& n, float k1, float k2) {
  auto r0 = (k1-k2)/(k1+k2);
  r0 *= r0;
  auto cosi = -dot(i, n);
  
  // Inside medium with higher refractive index:
  if (k1 > k2) {
    auto k = k1/k2;
    auto sin2t = k*k * (1.f - cosi*cosi);
    
    // And total internal reflectance (TIR):
    if (sin2t > 1) {
      return 1;
    }
  }
  
  // Inside medium with lower refractive index or
  // Inside medium with higher refractive index but below critical incidence angle (no TIR):
  auto x = (1.f-cosi);
  return r0 + (1.f - r0) * x * x * x * x * x;
}
