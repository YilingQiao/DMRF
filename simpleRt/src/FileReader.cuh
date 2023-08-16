#ifndef FILEREADER_CUH
#define FILEREADER_CUH

#include <vector>
#include "hittable.cuh"

using namespace std;

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
#endif

class FileReader {
private:
	static void read_obj_file(char *dir, vector<hittable*> &vec_obj_list, material *mat_ptr);
public:
	static bool readfile_to_render(
		vector<hittable*>& vec_obj_list, 
		vector<hittable*>& vec_lightsrc_list,
		vector<hittable*>& vec_geom_list,
		const char *path,          // 
		int &nx, int &ny, int &ns, //
		camera *&c                // 
		);
};

#endif