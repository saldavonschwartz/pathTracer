//
//  Scene.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Scene.hpp"
#include "HitDetection.hpp"

using namespace std;

shared_ptr<Hittable> generateSimpleScene() {
  HittableVector scene;
  scene.push_back(make_shared<Sphere>(vec3(0,0,-1), 0.5, make_shared<Diffuse>(vec3(0.1, 0.2, 0.5))));
  scene.push_back(make_shared<Sphere>(vec3(0,-100.5,-1), 100, make_shared<Diffuse>(vec3(0.8, 0.8, 0.0))));
  scene.push_back(make_shared<Sphere>(vec3(1,0,-1), 0.5, make_shared<Metal>(vec3(0.8, 0.6, 0.2), 0.3)));
  scene.push_back(make_shared<Sphere>(vec3(-1,0,-1), 0.5, make_shared<Dielectric>(1.5)));
  scene.push_back(make_shared<Sphere>(vec3(-1,0,-1), -0.45, make_shared<Dielectric>(1.5)));
  return make_shared<HittableVector>(scene);
}

shared_ptr<Hittable> generateComplexScene() {
  HittableVector scene;
  scene.push_back(make_shared<Sphere>(vec3{0.f,-1000.f,0.f}, 1000.f, make_shared<Diffuse>(vec3{0.5f})));
  
  for (int a = -10; a < 10; a++) {
    for (int b = -10; b < 10; b++) {
      auto materialProbability = urand(0, 1);
      vec3 center{a + 0.9f * urand(0, 1), 0.2f, b + 0.9f * urand(0, 1)};
      if (length(center - vec3{4.f, 0.2f, 0.f}) > 0.9f) {
        if (materialProbability < 0.8f) {
          // Diffuse
          auto albedo = sphericalRand(1.) * sphericalRand(1.);
          scene.push_back(make_shared<Sphere>(center, 0.2f, make_shared<Diffuse>(albedo)));
        }
        else if (materialProbability < 0.95f) {
          // Metal
          auto albedo = sphericalRand(.5f) + 0.5f;
          auto fuzziness = urand(0, 1);
          scene.push_back(make_shared<Sphere>(center, 0.2f, make_shared<Metal>(albedo, fuzziness)));
        } else {
          // glass
          scene.push_back(make_shared<Sphere>(center, 0.2f, make_shared<Dielectric>(1.5f)));
        }
      }
    }
  }
  
  scene.push_back(make_shared<Sphere>(vec3{0.f, 1.f, 0.f}, 1.f, make_shared<Dielectric>(1.5f)));
  scene.push_back(make_shared<Sphere>(vec3{-4.f, 1.f, 0.f}, 1.f, make_shared<Diffuse>(vec3{0.4f, 0.2f, 0.1f})));
  scene.push_back(make_shared<Sphere>(vec3{4.f, 1.f, 0.f}, 1.f, make_shared<Metal>(vec3{0.7f, 0.6f, 0.5f}, 0.f)));
  return make_shared<HittableVector>(scene);
}



