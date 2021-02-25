#ifndef __NEURAL_H
#define __NEURAL_H

int Create_Model(int layers,unsigned int *layer_nodes,double ***(parm[]));
void Set_Simulate
(double ***model,int pre_ly_index,
 int cur_ly_index, double ***layer,
 int pre_nds,int cur_nds,int layers);
void Back_Simulate
(double ***model,int pre_ly_index,
 int cur_ly_index, double ***gap,
 double **layer,int pre_nds,int cur_nds,int pre_pre_nds);
void Model_Simulate
(double ****model_gap,int pre_ly_index,
 int cur_ly_index, double **gap,
 double **layer,int pre_nds,int cur_nds);
void Gd_Simulate
(double ***model_gap,double ****model,
 int pre_ly_index,int cur_ly_index,
 int pre_nds,int cur_nds,int sample_nums);
void Test_Model(const char *file_route,unsigned int *layer_nodes,
unsigned int layers,double ***layer,double ***model,
char** class_names,unsigned int ct);
void Rate_Learn(unsigned int *layer_nodes,unsigned int tensor_count,double **data_set,
 double **layer,unsigned int layers,double ***nodes_parm,char** class_names,
 unsigned int N,int *labels,unsigned int ct);
#endif