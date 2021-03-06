//
//  Camera.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Camera.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

Camera::Camera(const vec3& position, const vec3& lookAt, float fovy, float aspectRatio, float focalLength, float aperture)
: position(position), fovy(fovy), aspectRatio(aspectRatio), focalLength(focalLength), aperture(aperture) {
  float hh = tan(glm::radians(fovy)/2.f);
  float hw = aspectRatio * hh;

  // [x y z p] = new camera orientation:
  vec3 z = normalize(position - lookAt);
  x = normalize(cross({0.f, 1.f, 0.f}, z));
  y = cross(z, x);

  auto& f = focalLength;
  lowerLeftImageOrigin = position - x*hw*f - y*hh*f - z*f;
  hOffset = 2*hh*f*y;
  wOffset = 2*hw*f*x;
}

Ray Camera::castRay(float u, float v) {
  auto r = (aperture / 2.f) * glm::diskRand(1.f);
  auto offset = x*r.x + y*r.y;
  auto dir = lowerLeftImageOrigin + u*wOffset + v*hOffset - position - offset;
  return {position + offset, dir};
}
