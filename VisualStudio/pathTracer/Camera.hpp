//
//  Camera.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Camera_hpp
#define Camera_hpp

#include "Ray.hpp"

class Camera {
public:
  vec3 position;
  float aspectRatio;
  float focalLength;
  float aperture;
  float fovy;

  Camera(const vec3& position, const vec3& lookAt, float fovy, float aspectRatio, float focalLength, float aperture);
  
  Ray castRay(float u, float v);

private:
  float hh, hw;
  vec3 x, y;
  vec3 hOffset;
  vec3 wOffset;
  vec3 lowerLeftImageOrigin;
};

#endif /* Camera_hpp */
