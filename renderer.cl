struct Ray{
	float3 origin;
	float3 dir;
};

struct Light{
	float3 color;
	float3 dir;
};

struct Shape{
	float3 scale;
	float3 pos; // left/right, up/down, forward/backward
	float3 color;
	int shape_type; // 0 = sphere, 1 = plane, 2 = cilinder
};

struct Ray createCamRay(const int x_coord, const int y_coord, const int width, const int height){
	float fx = (float)x_coord / (float)width;
	float fy = (float)y_coord / (float)height;
	float fx2 = (fx - 0.5f) * ((float)(width) / (float)(height));
	float fy2 = fy - 0.5f;
	float3 pixel_pos = (float3)(fx2, -fy2, 0.0f);
	struct Ray ray;
	ray.origin = (float3)(0.0f, 0.0f, 70.0f);
	ray.dir = normalize(pixel_pos - ray.origin);

	return ray;
}

bool intersect_shape(const struct Shape* shape, const struct Ray* ray, float* t) {
	float l = ray->dir.x;
	float m = ray->dir.y;
	float n = ray->dir.z;
	float x = ray->origin.x - shape->pos.x;
	float y = ray->origin.y - shape->pos.y;
	float z = ray->origin.z - shape->pos.z;

	if(shape->shape_type == 0) { // sphere
		float A = l * l + m * m + n * n;
		float B = 2.0f * (l * x + m * y + n * z);
		float C = x * x + y * y + z * z - shape->scale.x * shape->scale.x;

		float D = B * B - 4.0f * A * C;
		if (D < 0.0f) return false;

		float t1 = (-B - sqrt(D)) / (2.0f * A);
		//float t2 = (-B + sqrt(D)) / (2.0f * A);
		if (t1 > 0.0f && t1 < *t) {
			*t = t1;
			return true;
		}

	} else if(shape->shape_type == 1) { // inf plane
		float D = shape->pos.y;
		float A = 0;
		float B = 1;
		float C = 0;

		float t0 = (D - (A * ray->origin.x + B * ray->origin.y + C * ray->origin.z)) / (l * A + m * B + n * C);
		if (t0 > 0.0f && t0 < *t) {
			*t = t0;
			return true;
		}

	} else if(shape->shape_type == 2) { // cilinder
		float A = l * l + n * n;
		float B = 2.0f * (l * x + n * z);
		float C = x * x + z * z - shape->scale.x * shape->scale.x;

		float D = B * B - 4.0f * A * C;
		if (D < 0.0f) return false;

		float t1 = (-B - sqrt(D)) / (2.0f * A);
		//float t2 = (-B + sqrt(D)) / (2.0f * A);
		if (t1 > 0.0f && t1 < *t) {
			*t = t1;
			return true;
		}
	}
	
	return false;
}

__kernel void render_kernel(__global float3* output, int width, int height, float3 bg_color, __global const struct Shape* world_geomerty, int world_geo_size, struct Light sun)
{
	struct Shape shape = world_geomerty[2];
	//printf("shape %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %d\n", shape.scale.x, shape.scale.y, shape.scale.z, shape.pos.x, shape.pos.y, shape.pos.z, shape.color.x, shape.color.y, shape.color.z, shape.shape_type);
	const int gid = get_global_id(0);
	int x = gid % width;
	int y = gid / width;

	//float fx = (float)x / (float)width;
	//float fy = (float)y / (float)height;
	//output[gid] = (float3)(fx, fy, 0);

	struct Ray camray = createCamRay(x, y, width, height);

	float t = 1e20;
	float3 hit_color = bg_color;
	for(int i = 0; i < world_geo_size; i++) {
		struct Shape shape = world_geomerty[i];
		//printf("shape %.1f %.1f %.1f %.1f %.1f %.1f\n", shape.pos.x, shape.pos.y, shape.pos.z, shape.color.x, shape.color.y, shape.color.z);
		if (intersect_shape(&shape, &camray, &t)) {
			hit_color = shape.color;
		}
	}

	output[gid] = hit_color;

}