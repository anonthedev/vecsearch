#ifndef VECSEARCH_ASSIGN_H
#define VECSEARCH_ASSIGN_H

#ifdef __cplusplus
extern "C" {
#endif

int assign_upload(const float* vecs, int n, int dim);
int assign_run(const float* centers, const float* centerNorms, int* assign, int n, int k, int dim);
void assign_release(void);
const char* assign_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
