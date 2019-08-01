#include "rtree.h"
// typedef RTree<GPPointType *, float, 3, float>
// using std::vector;

int main() {
	RTree<float*, float, 3, float> myRtree;
	float p[] = {0.1, 0.3, 0.5};
	myRtree.Insert(p,p,0);
	return 0;
}