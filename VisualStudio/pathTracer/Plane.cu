//
//  Plane.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Plane.hpp"
#include "cuda_runtime.h"

__device__ Plane::Plane(const gvec3& position, float width, float height, Material* material) 
  : position(position), width(width), height(height), material(material) {
}

__device__ Plane::~Plane() {
  delete material;
}

__device__ bool Plane::boundingBox(double t0, double t1, AABA& bBox) const {
  gvec3 extent{ width / 2.f, height  / 2.f, 0.0001 };
  bBox = AABA(position - extent, position + extent);
  return true;
}

__device__ bool Plane::hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const {
  gvec3 x{ width /2.f, 0.f, position.z};
  gvec3 y{ 0.f, height / 2.f, position.z };
  auto n = normalize(cross(x, y));
  auto t = (dot(n, position) - dot(n, ray.origin)) / dot(n, ray.dir);
  
  if (tmin > t || t > tmax) {
    return false;
  }

  auto hitPoint = ray(t);
  
  if (length2(hitPoint - position) > length2(x + y)) {
    return false;
  }

  info.isFrontFace = dot(n, ray.dir) < 0.f;
  info.material = material;
  info.hitPoint = hitPoint;
  info.normal = n;
  info.t = t;
  
  return true;
}
