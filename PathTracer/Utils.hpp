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
#include <random>
#include <glm/vec3.hpp>

using glm::vec3;

const double inf = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

double urand(double min, double max);
vec3 to01Range(const vec3& v, bool normalize=false);
vec3 reflect(const vec3& i, const vec3& n);
vec3 refract(const vec3& i, const vec3& n, float k1, float k2);
float reflectance(const vec3& i, const vec3& n, float k1, float k2);

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
