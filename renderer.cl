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
	float diffuce;
	float sheen;
	float ambiant;
};

struct RayHit{
	float3 hit_pos;
	float3 normal;
	struct Shape obj;
	float distance;
	float shadow;
};

struct Ray createCamRay(const int x_coord, const int y_coord, const int width, const int height){
	struct Ray ray;

	ray.origin = (float3)(0.0f, 0.0f, 0.0f);
	float3 f = (float3)((float)width / (float)height, 1.0f, 0.5f);
	ray.dir = normalize((float3)(
		((float)x_coord + 0.5f) * f.x / (float)width - f.x * 0.5f,
		f.y * 0.5f - ((float)y_coord + 0.5f) * f.y / (float)height,
		f.z
	));

	return ray;
}

bool intersect_shape(const struct Shape* shape, const struct Ray* ray, struct RayHit* hit) {
	float l = ray->dir.x;
	float m = ray->dir.y;
	float n = ray->dir.z;
	float x = ray->origin.x - shape->pos.x;
	float y = ray->origin.y - shape->pos.y;
	float z = ray->origin.z - shape->pos.z;
	float t1 = 0.0f, t2 = 0.0f;
	float3 normal = (float3)(0.0f, 0.0f, 0.0f);

	if(shape->shape_type == 0) { // sphere
		float A = l * l + m * m + n * n;
		float B = 2.0f * (l * x + m * y + n * z);
		float C = x * x + y * y + z * z - shape->scale.x * shape->scale.x;

		float D = B * B - 4.0f * A * C;
		if (D <= 0.0f) return false;

		t1 = (-B - sqrt(D)) / (2.0f * A);
		t2 = (-B + sqrt(D)) / (2.0f * A);
		if (t1 > t2 && t2 > 0.0f) t1 = t2;
		normal = normalize(ray->origin + ray->dir * t1 - shape->pos);

	} else if(shape->shape_type == 1) { // inf plane
		float D = shape->pos.y;
		float A = 0;
		float B = 1;
		float C = 0;

		t1 = (D - (A * ray->origin.x + B * ray->origin.y + C * ray->origin.z)) / (l * A + m * B + n * C);
		normal = (float3)(0.0f, 1.0f, 0.0f);

	} else if(shape->shape_type == 2) { // cilinder
		float A = l * l + n * n;
		float B = 2.0f * (l * x + n * z);
		float C = x * x + z * z - shape->scale.x * shape->scale.x;

		float D = B * B - 4.0f * A * C;
		if (D <= 0.0f) return false;

		t1 = (-B - sqrt(D)) / (2.0f * A);
		t2 = (-B + sqrt(D)) / (2.0f * A);
		if (t1 > t2 && t2 > 0.0f) t1 = t2;

		float3 hit_pos = ray->origin + ray->dir * t1;
		float3 shape_pos = shape->pos;
		shape_pos.y = hit_pos.y;
		normal = normalize(ray->origin + ray->dir * t1 - shape_pos);
	}

	if (t1 > 0.0f && t1 < hit->distance) {
		*hit = (struct RayHit){
			ray->origin + ray->dir * t1,
			normal,
			(struct Shape){
				shape->scale,
				shape->pos,
				shape->color,
				shape->shape_type,
				shape->diffuce,
				shape->sheen,
				shape->ambiant
			},
			t1
		};
		return true;
	}
	
	return false;
}

__kernel void render_kernel(
	__global float3* output, int width, int height, float3 bg_color,
	__global const struct Shape* world_geomerty, int world_geo_size, struct Light sun,
	__global const float* params
) {
	float shadow_strength = params[0];
	float ambiant_strength = params[1];
	float sheen_strength = params[2];
	float diffuce_strength = params[3];
	float base_color_strength = params[4];

	const int gid = get_global_id(0);
	int x = gid % width;
	int y = gid / width;

	struct Ray camray = createCamRay(x, y, width, height);
	float t = 1e20;
	struct Shape temp;
	temp.color = bg_color;
	temp.diffuce = .5f;
	temp.sheen = 0.0f;
	temp.ambiant = 0.5f;
	struct RayHit hit = (struct RayHit){
		camray.origin + camray.dir * t,
		camray.dir,
		temp,
		t, 0
	};

	bool b = false;
	for(int i = 0; i < world_geo_size; i++) {
		struct Shape shape = world_geomerty[i];
		if (intersect_shape(&shape, &camray, &hit)) {
			b = true;
		}
	}
	// LAZY SHADOWS
	if (b) {
		struct Ray ray = (struct Ray){ hit.hit_pos + hit.normal*0.01f, -sun.dir };
		float t = 1e20;
		struct RayHit filler_hit;
		filler_hit.distance = t;

		for(int i = 0; i < world_geo_size; i++) {
			struct Shape shape = world_geomerty[i];
			if (intersect_shape(&shape, &ray, &filler_hit)) {
				hit.shadow = 1.0f;
				break;
			}
		}
	}

	// Phong lighting
	int f = 3;
	float3 r = sun.dir + dot(2, dot( dot(sun.dir,hit.normal), hit.normal));
	float3 v = normalize(camray.dir);
	float3 Ishe = sun.color * hit.obj.sheen * hit.obj.color * max(0.0f, pow(dot(r, v),f));
	float3 Idif = sun.color * hit.obj.diffuce * hit.obj.color * max(0.0f, dot(hit.normal, -sun.dir));
	float3 Iamb = sun.color * hit.obj.ambiant * hit.obj.color;
	float3 I = Idif*diffuce_strength + Ishe*sheen_strength + Iamb*ambiant_strength;

	output[gid] =  I + hit.obj.color*base_color_strength - (I*hit.shadow*shadow_strength*Idif);
}