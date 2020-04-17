//
//  Ray.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Ray_hpp
#define Ray_hpp

#include <glm/glm.hpp>

using glm::vec3;

class Ray {
public:
  vec3 origin;
  vec3 dir;

  Ray() = default;

  Ray(vec3 origin, vec3 dir)
  : origin(origin), dir(normalize(dir)) {}

  vec3 operator()(float t) const {
    return origin + t * dir;
  }
};


#endif /* Ray_hpp */
