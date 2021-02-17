#ifndef __NEURAL_H
#define __NEURAL_H

int Create_Model(int layers,unsigned int *layer_nodes,double ***(parm[]));
void Set_Simulate
(double ***model,int pre_ly_index,
 int cur_ly_index, double ***layer,
 int pre_nds,int cur_nds);
void Back_Simulate
(double ***model,int pre_ly_index,
 int cur_ly_index, double ***gap,
 double **layer,int pre_nds,int cur_nds);
void Model_Simulate
(double ****model_gap,int pre_ly_index,
 int cur_ly_index, double **gap,
 double **layer,int pre_nds,int cur_nds);
void Gd_Simulate
(double ***model_gap,double ****model,
 int pre_ly_index,int cur_ly_index,
 int pre_nds,int cur_nds,int sample_nums);
#endif