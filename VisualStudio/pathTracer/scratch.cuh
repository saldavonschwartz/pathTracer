//class Hittable {
//public:
//	struct HitInfo {
//		gvec3 hitPoint{ 0.f };
//		// ...
//	};
//
//	// ...
//	__device__ virtual ~Hittable() {};
//};
//
//class HittableVector : public Hittable {
//public:
//	Hittable** data = nullptr;
//	size_t size = 0;
//	size_t idx = 0;
//
//	__device__ HittableVector() {}
//
//	__device__ void init(size_t size) {
//		this->size = size;
//		this->data = new Hittable*[size];
//	}
//
//	__device__ void HittableVector::add(Hittable* hittable) {
//		data[idx] = hittable;
//		idx = (idx + 1) % size;
//	}
//};
//
//// case 1: Works:
//
//__global__ void generateSimpleScene(HittableVector** scene, Camera* cam, float aspect) {
//	// First I 'new' the scene. Then call its init which 'new's the internal data.
//	*scene = new HittableVector();
//	(*scene)->init(5);
//	(*scene)->add(new Sphere());
//	(*scene)->add(new Sphere());
//	(*scene)->add(new Sphere());
//	(*scene)->add(new Sphere());
//	(*scene)->add(new Sphere());
//
//	gvec3 lookFrom = { 3.f, 3.f, 2.f };
//	gvec3 lookAt = { 0.f, 0.f, -1.f };
//	float f = length(lookFrom - lookAt);
//	float a = 0.1f;
//	float fovy = 20.f;
//
//	// No need to 'new' the camera.
//	cam->init(lookFrom, lookAt, fovy, aspect, f, a);
//}
//
//__global__ void renderScene(
//	gvec3* frameBuff, int w, int h, Camera* cam, Hittable** scene,
//	int raysPerPixel, int maxBouncesPerRay, curandState* rStates
//)
//{
//	// ...
//
//	// ... A device func which takes Hittable* param
//	anotherDeviceFunc(*scene);
//}
//
//int main() {
//	// scene is ptr to ptr
//	HittableVector** scene;
//	cudaMalloc(&scene, sizeof(HittableVector));
//
//	// cam is just ptr
//	Camera* cam;
//	cudaMalloc(&cam, sizeof(Camera));
//
//	// both get initialized correctly and work
//	generateSimpleScene << <1, 1 >> > (scene, cam, aspect);
//	renderScene << <blocks, threads >> > (
//		frameBuff, width, height, cam, (Hittable**)scene, raysPerPixel, maxBouncesPerRay, perPixelRand
//	);
//}
//
//// case 2: scene is corrupt or undef or uninitialized... who knows...
//
//__global__ void generateSimpleScene2(HittableVector* scene, Camera* cam, float aspect) {
//	// main diff: I pass the scene same as I pass the camera. I don't 'new' the object.
//	// but the init does new the internal data.
//	scene->init(5);
//	(*scene)->add(new Sphere());
//	(*scene)->add(new Sphere());
//	(*scene)->add(new Sphere());
//	(*scene)->add(new Sphere());
//	(*scene)->add(new Sphere());
//
//	gvec3 lookFrom = { 3.f, 3.f, 2.f };
//	gvec3 lookAt = { 0.f, 0.f, -1.f };
//	float f = length(lookFrom - lookAt);
//	float a = 0.1f;
//	float fovy = 20.f;
//	// No need to 'new' the camera.
//	cam->init(lookFrom, lookAt, fovy, aspect, f, a);
//}
//
//__global__ void renderScene(
//	gvec3* frameBuff, int w, int h, Camera* cam, Hittable* scene,
//	int raysPerPixel, int maxBouncesPerRay, curandState* rStates
//)
//{
//	// ...
//
//	// ... A device func which takes Hittable* param
//	anotherDeviceFunc(scene);
//}
//
//int main() {
//	// scene is ptr this time (same as camera)
//	HittableVector* scene;
//	cudaMalloc(&scene, sizeof(HittableVector));
//
//	// cam is just ptr
//	Camera* cam;
//	cudaMalloc(&cam, sizeof(Camera));
//
//	generateSimpleScene << <1, 1 >> > (scene, cam, aspect);
//	renderScene << <blocks, threads >> > (
//		frameBuff, width, height, cam, scene, raysPerPixel, maxBouncesPerRay, perPixelRand
//	);
//
//	// crashes here: error 707: 
//	cudaDeviceSynchronize();
//}