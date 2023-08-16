#include "hittable.cuh"
#include "hittable_list.cuh"
#include "bvh.cuh"
#include "material.cuh"
#include "ray.cuh"

#define MAX_LEAF_OBJ 10
#define MAX_DEPTH 30
#define MAX_STACK 70

bvh_node::bvh_node() {
	type = type_bvh_node;
	mat_ptr = NULL;
	obj_list = NULL;
	left = NULL;
	right = NULL;
}


__device__ bool bvh_node::test(aabb &rec) {
	rec = box;
	return true;
}


bool bvh_node::bounding_box(float t0, float t1, aabb& b) {
	b = box;
	return true;
}

__device__ void bvh_node::refit() {
	if (left == NULL and right == NULL)
		obj_list->bounding_box(0, 1, box);
	else {
		((bvh_node*) left)->refit();
		((bvh_node*) right)->refit();

		aabb box_l, box_r;
        left->bounding_box(0, 1, box_l);
        right->bounding_box(0, 1, box_r);
        box = surrounding_box(box_l, box_r);
	}

}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	bool is_hit = false;
	// hittable **node_stack=new hittable*[MAX_DEPTH];
	// printf("bvh ~~~~~~~~~~ hit 0\n");
	hittable *node_stack[MAX_STACK];
	// printf("bvh ~~~~~~~~~~ hit 1\n");
	int top = 0;
	node_stack[++top] = (hittable*)this;
	rec.t = 1e18;
	// printf("bvh ~~~~~~~~~~ hit 2\n");
	while (top) {
		// printf("top %d \n", top);
		// if (top > MAX_DEPTH / 2)
		// 	printf("top %d ~~~~~~~~~~~~~~~~~~~~\n", top);
		bvh_node *p = (bvh_node*)node_stack[top--];
		if ((p->box).hit(r, t_min, rec.t)) { //p->box.hit(r, t_min, rec.t)
			//Leaf node
			if (p->left == NULL && p->right == NULL) {
				is_hit |= p->obj_list->hit(r, t_min, rec.t, rec);
			}
			else {
				node_stack[++top] = p->left;
				node_stack[++top] = p->right;
			}
		}
	}
	// delete[] node_stack;
	return is_hit;
	//return false;
}

__global__ void visit(bvh_node **root, int *result) {
	//bvh_node *tmp = *root;
	//while(tmp!=NULL)
	//	tmp = (bvh_node*)tmp->left;
	//tmp = (bvh_node*)tmp->right;
	hittable *node_stack[200];
	int top = 0;
	node_stack[++top] = (hittable*)(*root);
	int tot = 0;
	while (top) {
		bvh_node *p = (bvh_node*)node_stack[top--];
		if (p->left == NULL && p->right == NULL) {
			hittable_list* lst = p->obj_list;
			int lst_size = lst->list_size;
			tot += lst_size;
		}
		else {
			node_stack[++top] = p->left;
			node_stack[++top] = p->right;
		}
	}
	*result = tot;
}

hittable *tmp[1000000];

void split_objs(hittable **l, int &pl, int L, int R, float split, int axis) {
	pl = L;
	int pr = R - 1;
	//if (L == 15 && R == 18) {
	//	pl--;
	//	pl++;
	//}
	for (int i = L; i < R; i++) {
		if (l[i]->center[axis] < split) {
			tmp[pl++] = l[i];
		}
		else {
			tmp[pr--] = l[i];
		}
	}
	for (int i = L; i < R; i++)
		l[i] = tmp[i];
}

float calc_sah(hittable **l, int mid, int L, int R) {
	aabb box_l, box_r, tmp;
	l[L]->bounding_box(0, 1, box_l);
	l[R - 1]->bounding_box(0, 1, box_r);
	for (int i = L; i < mid; i++) {
		l[i]->bounding_box(0, 1, tmp);
		box_l = surrounding_box(box_l, tmp);
	}
	for (int i = mid; i < R - 1; i++) {
		l[i]->bounding_box(0, 1, tmp);
		box_r = surrounding_box(box_r, tmp);
	}
	float area_l = (box_l.max()[0] - box_l.min()[0])*(box_l.max()[1] - box_l.min()[1]) +
		(box_l.max()[0] - box_l.min()[0])*(box_l.max()[2] - box_l.min()[2]) +
		(box_l.max()[1] - box_l.min()[1])*(box_l.max()[2] - box_l.min()[2]);

	float area_r = (box_r.max()[0] - box_r.min()[0])*(box_r.max()[1] - box_r.min()[1]) +
		(box_r.max()[0] - box_r.min()[0])*(box_r.max()[2] - box_r.min()[2]) +
		(box_r.max()[1] - box_r.min()[1])*(box_r.max()[2] - box_r.min()[2]);

	return (mid - L) * area_l / (area_l + area_r) + (R - mid) * area_r / (area_l + area_r);
}

bvh_node::bvh_node(hittable **l, int L, int R, float time0, float time1, int depth) {
	int n = R - L;
	type = type_bvh_node;
	// if (depth >= 17)
	// 	printf("L:%d R:%d depth:%d MAX_DEPTH:%d n:%d MAX_LEAF_OBJ:%d\n", L, R, depth, MAX_DEPTH, n, MAX_LEAF_OBJ);
	if (depth >= MAX_DEPTH || n <= MAX_LEAF_OBJ) {
		mat_ptr = NULL;
		obj_list = new hittable_list(l + L, R - L);
		obj_list->bounding_box(0, 1, box);
		left = right = NULL;
	}
	else {
		obj_list = NULL;
		mat_ptr = NULL;
		//hittable **tmp = (hittable**)malloc(n * sizeof(hittable*));
		l[L]->bounding_box(0, 1, box);
		for (int i = L + 1; i < R; i++) {
			aabb box2;
			l[i]->bounding_box(0, 1, box2);
			box = surrounding_box(box, box2);
		}
		float min_sah = 1e9;
		float best_split;
		int best_axis = -1;
		//if (R == 18 && L == 15)
		//	min_sah = 1e10;
		for (int axis = 0; axis < 3; axis++) {
			float lowerbound = box.min()[axis], upperbound = box.max()[axis];
			float step = (upperbound - lowerbound) / (32.0f);
			for (float split = lowerbound + step; split < upperbound - 1e-5f; split += step) {
				int mid;
				split_objs(l, mid, L, R, split, axis);
				if (mid == L || mid == R)
					continue;
				float now_sah = calc_sah(tmp, mid, L, R);
				if (now_sah < min_sah) {
					min_sah = now_sah;
					best_axis = axis;
					best_split = split;
				}
			}
		}
		if (best_axis == -1) {
			std::cerr << "Can't find the best split!\n";
			exit(999);
		}
		else {
			int mid;
			split_objs(l, mid, L, R, best_split, best_axis);
			//free(tmp);

			left = new bvh_node(l, L, mid, time0, time1, depth + 1);
			right = new bvh_node(l, mid, R, time0, time1, depth + 1);
		}

		aabb box_l, box_r;
		left->bounding_box(0, 1, box_l);
		right->bounding_box(0, 1, box_r);
		box = surrounding_box(box_l, box_r);
	}
}

__device__ vec3 bvh_node::sample(curandState *local_rand_state) const {
	return vec3(0., 100., 0.);
	// int idx = 0;
	// return obj_list->sample(local_rand_state);
}

