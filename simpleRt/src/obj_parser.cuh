#ifndef OBJ_PARSER_CUH
#define OBJ_PARSER_CUH

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "vec3.cuh"
#include "hittable.cuh"
#include "triangle.cuh"
#include "material.cuh"

using namespace std;

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
#endif

// void read_obj_file(char *dir, hittable** &list, int &list_size) {
// 	FILE *f;
// 	fopen_s(&f, dir, "r");
// 	if (f == NULL) {
// 		exit(9);
// 	}
// 	vector<vec3> points, p_normals;
// 	vector<hittable*> vec_objs;
// 	char line[100];
// 	int obj_id = 0;
// 	if (f != NULL) {
// 		while (fgets(line, 100, f) != NULL) {
// 			int p = 0;
// 			while ((line[p] >= 'a'&&line[p] <= 'z') || (line[p] >= 'A'&&line[p] <= 'Z')) p++;
// 			if (line[0] == 'o') {
// 				obj_id++;
// 			}
// 			/*else if (line[0] == 'v'&&line[1] != 'n') {
// 				double x, y, z;
// 				sscanf_s(line + p, "%lf%lf%lf", &x, &y, &z);
// 				vec3 temp = vec3(x, y, z);
// 				//cout<<x*500<<' '<<y*500<<' '<<z*500<<endl;
// 				points.push_back(temp);
// 			}
// 			else if (line[0] == 'v'&&line[1] == 'n') {
// 				double x, y, z;
// 				//has_normal = true;
// 				sscanf_s(line + p, "%lf%lf%lf", &x, &y, &z);
// 				vec3 temp = vec3(x, y, z);
// 				//cout<<x*500<<' '<<y*500<<' '<<z*500<<endl;
// 				p_normals.push_back(temp);
// 			}
// 			else if (line[0] == 'f') {
// 				int x, y, z;
// 				sscanf_s(line + p, "%d%d%d", &x, &y, &z);
// 				//list[list_size++]
// 				if(obj_id==1)
// 					vec_objs.push_back(new triangle(points[x - 1], points[y - 1], points[z - 1],
// 						//p_normals[x - 1], p_normals[y - 1], p_normals[z - 1],
// 						new metal(vec3(0.7, 0.6, 0.5), 0.0)));
// 				else if(obj_id==3)
// 					vec_objs.push_back(new triangle(points[x - 1], points[y - 1], points[z - 1],
// 						new metal(vec3(0.7, 0.8, 0.2), 0.5)));
// 				else if (obj_id == 2)
// 					vec_objs.push_back(new triangle(points[x - 1], points[y - 1], points[z - 1],
// 						new dielectric(1.5)));
// 				else if (obj_id == 4)
// 					vec_objs.push_back(new triangle(points[x - 1], points[y - 1], points[z - 1],
// 						new lambertian(vec3(0.8, 0.8, 0.0))));
// 				//new lambertian(vec3(0.8, 0.8, 0.0)));*/
// 			}
// 		}
// 		list_size = vec_objs.size();
// 		list = (hittable**)malloc(list_size*sizeof(hittable*));
// 		for (int i = 0; i < list_size; i++) {
// 			list[i] = vec_objs[i];
// 		}
// 		fclose(f);
// }


#endif // !OBJ_PARSER_H
