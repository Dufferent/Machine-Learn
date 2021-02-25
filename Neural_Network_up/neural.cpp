#include "neural.h"
#include "stdio.h"
#include "stdlib.h"
#include "unistd.h"
#include "fcntl.h"
#include "sys/types.h"
#include "string.h"
#include "time.h"
#include "math.h"
#include "libgen.h"

#include "world.hpp"
#include "highgui.hpp"
#include "video.hpp"
#include "videoio.hpp"
#include "imgproc.hpp"
using namespace cv;

#include "iostream"
using namespace std;

typedef char** chpp;
// MAX_DATA_ITEM 的值限制了最大能获取训练数据的数据项数量
#define MAX_DATA_ITEM 8192
#define MAX_CHAR_LEN  64

#define IMG_WIDTH 182
#define IMG_HEIGHT 182

#define PREWITT_VERT 1
#define PREWITT_HORI 2
#define SOBEL_VERT 3
#define SOBEL_HORI 4
#define SCHARR_VERT 5
#define SCHARR_HORI 6

#define Fliter_Nums 6
#define Final_Size  15

#define A 0.02
#define G 0.9997
#define M 80
/*
 * @parm::layers 神经网络的层数 
 * @parm::layer_nodes 每一层神经网络的节点个数->layer_nodes[0]->输入层的节点个数
 * @parm::nodes_parm 模型训练完成后返回每一层每一个节点对应下一层的 权重 K-W 训练的参数值
 * @parm::data_set 用来训练神经网络的数据集(张量表示)
 * ......
*/
void CHECK_NODE(int layers,unsigned int *layer_nodes,double **layer);
chpp Get_Data_Name(int &ct,char data_root[]);
Mat Filiter(Mat src,int f,int s,int flags,int size);
unsigned char* Extend_Tensor(Mat *img_data,int nums);
void Clear_Model_DGap(double ****model_gap,int layers,unsigned int *layer_nodes);
Mat Polling_Layer(Mat src,int ly_sz);

int Create_Model(int layers,unsigned int *layer_nodes,double ***(parm[]))
{
    double **data_set;
    double ***nodes_parm = (*parm);
    //--随机数--
    srand((unsigned int)time(NULL));
    /* step1::=> 构建层级数组 */
    double **layer;
    layer = (double**)malloc(sizeof(double*)*layers);
    // printf  ("DEBUG::节点参数一览!\r\n");
    for (int i=0;i<layers;i++)
    {
        layer[i] = (double*)malloc(sizeof(double)*layer_nodes[i]);
        memset(layer[i],0,layer_nodes[i]*sizeof(double));
        // printf ("LAYER<%d>:\n",i+1);
        // for (int j=0;j<layer_nodes[i];j++)
        //     printf ("NODES<%d>: => VAL[%0.3lf]\n",j+1,layer[i][j]);
    }
    // printf  ("DEBUG::节点参数一览!\r\n");
    // printf  ("DEBUG::训练参数一览!\r\n");
    // for (int i=0;i<layers-1;i++)
    // {
    //     for (int j=0;j<layer_nodes[i];j++)
    //     {
    //         for (int k=0;k<layer_nodes[i+1];k++)
    //             printf ("node=><ly>[%d]<nd>[%d]<dst>[%d]::=>parm::[%lf]\r\n"
    //             ,i+1,j+1,k+1,
    //             nodes_parm[i][j][k]);
    //     }
    // }
    // printf  ("DEBUG::训练参数一览!\r\n");
    /* step2::=> 构建每一层节点对应的拟合函数(第一层没有映射源) */
    // for (int i=1;i<layers;i++)
    // {
    //     Set_Simulate(nodes_parm,i-1,i,
    //     &layer,layer_nodes[i-1],layer_nodes[i],layers
    //     );
    // }
    /* step3::=> 反向传播,代价函数 */
    //--处理样本--
    // char data_root[128] = {0};
    // printf ("请输入训练数据的(绝对/相对)路径::=>()\r\n");
    // scanf  ("%s",data_root);
    // printf ("您选择的路径是::[%s]\r\n",data_root);
    // int pfd[2];
    // int backfd;
    // char cmd[1024]={0};
    // char buf[1024]={0};
    // char **class_names;
    // int index = 0;
    // int ct = 0;
    // pipe(pfd);
    // backfd = dup(STDOUT_FILENO);    //备份标准输入输出流
    // dup2(pfd[1],STDOUT_FILENO);     //将标准输入输出重定向到管道
    // sprintf (cmd,"%s%s","ls ",data_root);
    // system(cmd);
    // read(pfd[0],buf,1024);          //读管道
    // dup2(backfd,STDOUT_FILENO);     //恢复标准输入输出
    // strncat (buf,"##",2);            //添加结束符
    // printf  ("[class_group]::%s",buf);
    // class_names = (char**)malloc(sizeof(char*)*32);
    // for (int i=0;i<32;i++)class_names[i]=(char*)malloc(sizeof(char)*32);
    // while (buf[index] != '#')
    // {
    //     if (buf[index] != '\n')
    //         strncat(class_names[ct],&buf[index],1);
    //     else
    //         ct++;
    //     index++;
    // }
    // for (int i=0;i<ct;i++)
    //     printf ("process class_names::=>[%s]\r\n",class_names[i]);
    char data_root[MAX_CHAR_LEN] = {0};
    printf ("请输入训练数据的(绝对/相对)路径::=>()\r\n");
    scanf  ("%s",data_root);
    printf ("您选择的路径是::[%s]\r\n",data_root);
    int ct = 0;
    char **class_names = Get_Data_Name(ct,data_root);
    printf ("class_name:=>[\r\n");
    for (int i=0;i<ct;i++)
        printf ("(%s),\r\n",class_names[i]);
    printf ("]\r\n");
    //--获取图像集的绝对路径--
    // printf ("img-dir::=>\r\n");
    int *img_count;
    chpp *img_name;
    img_count = (int*)malloc(sizeof(int)*ct);
    img_name  = (chpp*)malloc(sizeof(chpp)*ct);
    for (int i=0;i<ct;i++)img_count[i]=0;   //清空图片计数器
    for (int i=0;i<ct;i++)
    {
        char data_img[MAX_CHAR_LEN*2]={0};
        sprintf (data_img,"%s%c%s%c",data_root,'/',class_names[i],'\0');
        img_name[i] = Get_Data_Name(img_count[i],data_img);
        // for (int j=0;j<img_count[i];j++)
        //     printf ("%s\r\n",img_name[i][j]);
    }
    chpp *img_dir;
    img_dir = (chpp*)malloc(sizeof(chpp)*ct);
    for (int i=0;i<ct;i++)img_dir[i] = (chpp)malloc(sizeof(char*)*img_count[i]);
    for (int i=0;i<ct;i++)
        for (int j=0;j<img_count[i];j++)
            img_dir[i][j] = (char*)malloc(sizeof(char)*MAX_CHAR_LEN);
    for (int i=0;i<ct;i++)
        for (int j=0;j<img_count[i];j++)
        {
            char data_img[MAX_CHAR_LEN]={0};
            sprintf (data_img,"%s%c%s",data_root,'/',class_names[i]);
            sprintf (img_dir[i][j],"%s%c%s",data_img,'/',img_name[i][j]);
            // printf  ("%s\r\n",img_dir[i][j]); 
        }
    for (int i=0;i<ct;i++)
        printf ("获得到<类别::[%s]>图片张数:=>[%d]\r\n",class_names[i],img_count[i]);
    //--处理所有图片--
    // @layers 层数
    // @layer  节点数值
    // @nodes_parm 训练参数
    // @set_data 张量 (原始数据)
    // @ct 类别个数
    // @class_names 类别名称
    // @img_dir 图片绝对路径
    // @img_count 各类别图片数量
    int count = 0;
    for (int i=0;i<ct;i++)
        count += img_count[i];
    printf ("图片总张数::=>[%d]\r\n",count);
    printf ("请等待处理图片!\r\n");
    // @count 图片总张数
    unsigned char **tensor = (unsigned char**)malloc(sizeof(unsigned char*)*count);
    int tensor_count = 0;
    for (int i=0;i<ct;i++)
    {
        for (int j=0;j<img_count[i];j++)
        {
            Mat tmps = imread(img_dir[i][j]);
            if (!tmps.empty())
            {
                resize(tmps,tmps,Size(IMG_WIDTH,IMG_HEIGHT),0,0,INTER_LINEAR);
                cvtColor(tmps,tmps,CV_RGB2GRAY);
                Mat newtmps[Fliter_Nums];
                newtmps[0] = Filiter(tmps,3,2,PREWITT_HORI,IMG_HEIGHT);
                newtmps[1] = Filiter(tmps,3,2,PREWITT_VERT,IMG_HEIGHT);
                newtmps[2] = Filiter(tmps,3,2,SCHARR_HORI,IMG_HEIGHT);
                newtmps[3] = Filiter(tmps,3,2,SCHARR_VERT,IMG_HEIGHT);
                newtmps[4] = Filiter(tmps,3,2,SOBEL_HORI,IMG_HEIGHT);
                newtmps[5] = Filiter(tmps,3,2,SOBEL_VERT,IMG_HEIGHT);
                newtmps[0] = Polling_Layer(newtmps[0],6);
                newtmps[1] = Polling_Layer(newtmps[1],6);
                newtmps[2] = Polling_Layer(newtmps[2],6);
                newtmps[3] = Polling_Layer(newtmps[3],6);
                newtmps[4] = Polling_Layer(newtmps[4],6);
                newtmps[5] = Polling_Layer(newtmps[5],6);              
                //--图片数据::[FINAL_SIZE^2xFliter_Nums]--
                //--展开图片特征数据--
                // imshow("view",newtmps[0]);
                // waitKey(0);
                tensor[tensor_count] = Extend_Tensor(newtmps,Fliter_Nums);
                tensor_count++;
            }
        }
    }
    printf ("tensor[0]::[\r\n");
    for (int i=500;i<550;i++)  //  打印第一张图片十个张量数据
        printf ("%2d \r\n",tensor[0][i]);
    printf ("...]\r\n");
    //--@parm::=>tensor 图片数据的张量
    //--处理tensor=>(double)[0,1]
    data_set = (double**)malloc(sizeof(double*)*tensor_count);
    for (int i=0;i<tensor_count;i++)
        data_set[i] = (double*)malloc(sizeof(double)*Final_Size*Final_Size*Fliter_Nums);
    for (int i=0;i<tensor_count;i++)
        for (int j=0;j<Final_Size*Final_Size*Fliter_Nums;j++)
            data_set[i][j] = tensor[i][j] / 255.0;
    //--开始训练--
    printf ("data_set[0]::[\r\n");
    for (int i=500;i<550;i++)  //  打印第一张图片十个张量数据
        printf (" %0.3lf \r\n",data_set[0][i]);
    printf ("...]\r\n");
    printf ("请输入训练时间:=>\r\n");
    unsigned int times;
    scanf  ("%d",&times);
    unsigned int all = times;
    /* start transform */  
    double **gap = (double**)malloc(sizeof(double*)*layers); //全连接层没有误差
    for (int i=0;i<layers;i++)
    {
        gap[i] = (double*)malloc(sizeof(double)*layer_nodes[i]);
        memset(gap[i],0,layer_nodes[i]*sizeof(double));
        // printf ("LAYER<%d>:\n",i+2);
        // for (int j=0;j<layer_nodes[i];j++)
        //     printf ("NODES<%d>: => GAPS[%0.3lf]\n",j+1,gap[i][j]);
    }
    int *labels = (int*)malloc(sizeof(int)*tensor_count);
    for (int i=0;i<ct;i++){
        static int index = 0;
        static int cls_nm = 0;
        for (int j=0;j<img_count[i];j++)
            labels[index++] = cls_nm;
        cls_nm++;
    }
    //--参数误差偏导数--
    double ***model_gap = (double***)malloc(sizeof(double**)*layers); //第一维:=>层级;
    for (int i=0;i<layers;i++)                          //第二维:=>节点
        model_gap[i] = (double**)malloc(sizeof(double*)*layer_nodes[i]);
    for (int i=0;i<layers-1;i++)                        //第三维:=>权重(参数)
        for (int j=0;j<layer_nodes[i];j++)
        {
            model_gap[i][j] = (double*)malloc(sizeof(double)*layer_nodes[i+1]);
            memset(model_gap[i][j],0,sizeof(double)*layer_nodes[i+1]);
        }
    //--真实输出--
    double *real_out = (double*)malloc(sizeof(double)*layer_nodes[layers-1]);
    //--总代价--
    double J = 0.0;
    while (times--)
    {
        //--填充全连接层-- FINAL_SIZE^2xFliter_Nums => 每张图片的features => 全连接到第一层神经元上
        int uses = 0;
        int sample = (random()%tensor_count);
        while(1)
        {
            sample = (sample + (random()%tensor_count))%tensor_count;
            for (int i=0;i<layer_nodes[0];i++)
                layer[0][i] = data_set[sample][i];
            //--正向传播--
            for (int i=1;i<layers;i++)
            {
                Set_Simulate(nodes_parm,i-1,i,
                &layer,layer_nodes[i-1],layer_nodes[i],layers);
            }
            //--得到了每一层对应的值--
            //--反向传播--
            for (int i=0;i<layer_nodes[layers-1];i++)
            {
                if (i==labels[sample])real_out[i]=1;
                else real_out[i]=0;
            }
            // printf ("Tensor<%d>th ::=> out[car::%lf people::%lf]\r\n",sample,real_out[0],real_out[1]);
            //--逐层计算误差--
            for (int i=0;i<layer_nodes[layers-1];i++)   // SOFTMAX层的误差先计算出来
                gap[layers-1][i] = layer[layers-1][i] - real_out[i];
            for (int i=layers-1;i>1;i--)
            {
                Back_Simulate(nodes_parm,i-1,i,
                &gap,layer,layer_nodes[i-1],layer_nodes[i],layer_nodes[i-2]
                );
            }
            // sleep(2);
            //--计算误差偏导数--
            for (int i=1;i<layers;i++)
                Model_Simulate(&model_gap,i-1,i
                ,gap,layer,layer_nodes[i-1],layer_nodes[i]);
            //真实采用样本数
            uses++;
            if (uses > M)break;

            for (int i=0;i<ct;i++)
                J += ((real_out[i]*log(layer[layers-1][i]) +
                    (1-real_out[i])*log(1-layer[layers-1][i]))*(-1)/M);
        }
        //--梯度下降--
        for (int i=1;i<layers;i++)
            Gd_Simulate(model_gap,&nodes_parm,i-1,i,
            layer_nodes[i-1],layer_nodes[i],uses);
        //--清空误差--
        Clear_Model_DGap(&model_gap,layers,layer_nodes);
        //--计算总代价方便调试--
        // printf ("代价J::=>[%0.6lf]\r\n",J);
        system("clr.sh");
        printf ("TRANSING...\r\n");
        printf ("PROG::=> [");
        for (int i=0;i<(all-times)/1.0/all*30;i++)
            printf("#");
        printf ("]<%d%c>\r\n",(int)((all-times)/1.0/all*100),'%');
        printf ("J::=> [%0.6lf]\r\n",J);
        J=0;
    }
    Test_Model("./people.jpg",layer_nodes,layers,
    &layer,nodes_parm,class_names,ct);
    Test_Model("./car.jpg",layer_nodes,layers,
    &layer,nodes_parm,class_names,ct);
    return ct;
}

void Set_Simulate
(double ***model,int pre_ly_index,
 int cur_ly_index, double ***layer,
 int pre_nds,int cur_nds,int layers)
{
    // printf ("DEBUG::=>更新第<%d>层->第<%d>层的正向传播结果...\r\n",pre_ly_index,cur_ly_index);
    // printf ("DEBUG::=>pre_ly_index::[%d] pre_nds::[%d]\r\n",pre_ly_index,pre_nds);
    // printf ("DEBUG::=>cur_ly_index::[%d] cur_nds::[%d]\r\n",cur_ly_index,cur_nds);
    double **ly = *(layer);
    double *pre_ly = ly[pre_ly_index];
    double *cur_ly = ly[cur_ly_index];
    for (int i=0;i<cur_nds;i++)
    {   
        double tmps = 0;
        for (int j=0;j<pre_nds;j++)
        {
            tmps += (pre_ly[j]*(model[pre_ly_index][j][i]));  
            // printf ("DEBUG::i::[%d]\tj::[%d]...\r\n",i,j);
        }
        cur_ly[i] = 1/(1+exp((-1)*tmps));   //g(x)=>激活函数
        // cur_ly[i] = MAX(0,tmps);               //g(x)=>激活函数
        // printf ("DEBUG::=>cur_ly[%d]::VAL::[%lf]\r\n",i,cur_ly[i]);
    }
    // printf ("DEBUG::=>更新完毕...\r\n");
}

void CHECK_NODE(int layers,unsigned int *layer_nodes,double **layer)
{
    for (int i=0;i<layers;i++)
    {
        printf ("LAYER<%d>:\n",i+1);
        for (int j=0;j<layer_nodes[i];j++)
            printf ("NODES<%d>: => VAL[%0.3lf]\n",j+1,layer[i][j]);
    }
}

chpp Get_Data_Name(int &ct,char data_root[])
{
    int pfd[2];
    int backfd;
    char cmd[MAX_CHAR_LEN*2]={0};
    char buf[MAX_DATA_ITEM]={0};
    char **class_names;
    int index = 0;
    // int ct = 0;
    pipe(pfd);
    backfd = dup(STDOUT_FILENO);    //备份标准输入输出流
    dup2(pfd[1],STDOUT_FILENO);     //将标准输入输出重定向到管道
    sprintf (cmd,"%s%s","ls ",data_root);
    system(cmd);
    read(pfd[0],buf,MAX_DATA_ITEM);          //读管道
    dup2(backfd,STDOUT_FILENO);     //恢复标准输入输出
    strncat (buf,"#\0",2);            //添加结束符
    // printf  ("[class_group]::%s\r\n",buf);
    class_names = (char**)malloc(sizeof(char*)*MAX_DATA_ITEM);//最大支持8192个数据项
    for (int i=0;i<MAX_DATA_ITEM;i++)class_names[i]=(char*)malloc(sizeof(char)*MAX_CHAR_LEN);//项目名称最大64字节
    while (buf[index] != '#')
    {
        static int memct = 0;
        if (buf[index] != '\n')
        {
            class_names[ct][memct] = buf[index];
            memct++;
        }
        else
        {
            class_names[ct][memct] = '\0';
            memct = 0;
            ct++;
        }
        index++;
    }
    // for (int i=0;i<ct;i++)
    //     printf ("process class_names::=>[%s]\r\n",class_names[i]);
    return class_names;
}

Mat Filiter(Mat src,int f,int s,int flags,int size) //输入必须是灰度图
{
    //--创建滤波内核--
    int out_size = (size-f)/s+1;
    Mat out(Size(out_size,out_size),CV_8UC1,Scalar(0));
    char kernal[f][f];
    switch(flags)
    {
        case PREWITT_HORI:
        for (int i=0;i<f;i++)
        for (int j=0;j<f;j++)
        {
            if (i==0)kernal[i][j]=1;
            else if (i==1)kernal[i][j]=0;
            else kernal[i][j]=-1;
        }
        break;
        case PREWITT_VERT:
        for (int i=0;i<f;i++)
        for (int j=0;j<f;j++)
        {
            if (j==0)kernal[i][j]=1;
            else if (j==1)kernal[i][j]=0;
            else kernal[i][j]=-1;
        }
        break;
        case SCHARR_HORI:
        for (int i=0;i<f;i++)
        for (int j=0;j<f;j++)
        {
            if (i==0)
            {
                if (j==1)
                    kernal[i][j]=10;
                else
                    kernal[i][j]=3;
            }
            else if (i==1)kernal[i][j]=0;
            else {
                if (j==1)
                    kernal[i][j]=-10;
                else
                    kernal[i][j]=-3;
            }
        }
        break;
        case SCHARR_VERT:
        for (int i=0;i<f;i++)
        for (int j=0;j<f;j++)
        {
            if (j==0)
            {
                if (i==1)
                    kernal[i][j]=10;
                else
                    kernal[i][j]=3;
            }
            else if (j==1)kernal[i][j]=0;
            else {
                if (i==1)
                    kernal[i][j]=-10;
                else
                    kernal[i][j]=-3;
            }
        }
        break;
        case SOBEL_HORI:
        for (int i=0;i<f;i++)
        for (int j=0;j<f;j++)
        {
            if (i==0)
            {
                if (j==1)
                    kernal[i][j]=2;
                else
                    kernal[i][j]=1;
            }
            else if (i==1)kernal[i][j]=0;
            else {
                if (j==1)
                    kernal[i][j]=-2;
                else
                    kernal[i][j]=-1;
            }
        }
        break;
        case SOBEL_VERT:
        for (int i=0;i<f;i++)
        for (int j=0;j<f;j++)
        {
            if (j==0)
            {
                if (i==1)
                    kernal[i][j]=2;
                else
                    kernal[i][j]=1;
            }
            else if (j==1)kernal[i][j]=0;
            else {
                if (i==1)
                    kernal[i][j]=-2;
                else
                    kernal[i][j]=-1;
            }
        }
        break;
    }
    /*
    printf ("DEBUG::KERNAL_DATA_CHECK!\r\n");
    for (int i=0;i<f;i++)
    {
        for (int j=0;j<f;j++)
            printf (" %-2d",kernal[i][j]);
        printf ("\r\n");
    }*/
    for (int y=0;y<out_size;y+=s)
    for (int x=0;x<out_size;x+=s)
    {
        for (int i=0;i<f;i++)
        for (int j=0;j<f;j++)
            out.at<uchar>(Point(x,y)) += (int)(src.at<uchar>(Point(x+i,y+j))*kernal[i][j]);
    }

    return out;
}

unsigned char* Extend_Tensor(Mat *img_data,int nums)
{
    unsigned char *tensor;
    int data_count = 0;
    tensor = (unsigned char*)malloc(sizeof(unsigned char)*img_data[0].rows*img_data[0].cols*nums);
    for (int i=0;i<nums;i++)
    {
        for (int y=0;y<img_data[i].rows;y++)
        for (int x=0;x<img_data[i].cols;x++)
            tensor[data_count++] = img_data[i].at<uchar>(Point(x,y));
    }
    return tensor;
}

void Back_Simulate
(double ***model,int pre_ly_index,
 int cur_ly_index, double ***gap,
 double **layer,int pre_nds,int cur_nds,int pre_pre_nds)
{
    // printf ("DEBUG::=>更新第<%d>层->第<%d>层的反向传播结果...\r\n",cur_ly_index,pre_ly_index);
    // printf ("DEBUG::=>pre_ly_index::[%d] pre_nds::[%d]\r\n",pre_ly_index,pre_nds);
    // printf ("DEBUG::=>cur_ly_index::[%d] cur_nds::[%d]\r\n",cur_ly_index,cur_nds);
    double **ly = *(gap);
    double *pre_ly = ly[pre_ly_index];
    double *cur_ly = ly[cur_ly_index];
    for (int i=0;i<pre_nds;i++)
    {   
        double tmps = 0;
        for (int j=0;j<cur_nds;j++)
        {
            tmps += (cur_ly[j]*model[pre_ly_index][i][j]);  
            // printf ("DEBUG::i::[%d]\tj::[%d]...\r\n",i,j);
        }
        pre_ly[i] = tmps*layer[pre_ly_index][i]*(1-layer[pre_ly_index][i]);   //计算节点误差
        // double z = 0;
        // for (int j=0;j<pre_pre_nds;j++)
        //     z += (layer[pre_ly_index-1][j]*model[pre_ly_index-1][j][i]);
        // if (z<0)
        //     pre_ly[i] = tmps*0;
        // else
        //     pre_ly[i] = tmps*1;
        // pre_ly[i] = tmps*layer[pre_ly_index][i]*(1-layer[pre_ly_index][i]);
        // if (pre_ly_index != 0)
        //     printf ("DEBUG::=>pre_ly[%d]::VAL::[%0.16lf]\r\n",i,pre_ly[i]);
    }
    // printf ("DEBUG::=>更新完毕...\r\n");
}

void Model_Simulate
(double ****model_gap,int pre_ly_index,
 int cur_ly_index, double **gap,
 double **layer,int pre_nds,int cur_nds)
{
    // printf ("DEBUG::=>更新第<%d>层->第<%d>层的代价函数偏导计算结果...\r\n",pre_ly_index,cur_ly_index);
    // printf ("DEBUG::=>pre_ly_index::[%d] pre_nds::[%d]\r\n",pre_ly_index,pre_nds);
    // printf ("DEBUG::=>cur_ly_index::[%d] cur_nds::[%d]\r\n",cur_ly_index,cur_nds);
    double ***Dgap = (*model_gap);
    double **pre_Dgap = Dgap[pre_ly_index];
    double **cur_Dgap = Dgap[cur_ly_index];
    for (int i=0;i<pre_nds;i++)
        for (int j=0;j<cur_nds;j++)
        {
            pre_Dgap[i][j] += (layer[pre_ly_index][i]*gap[cur_ly_index][j]);
            // if (pre_ly_index != 0)
            //     printf ("DEBUG::=>DGAP<%d><%d>2<%d>::[%lf]\r\n",pre_ly_index,i,j,pre_Dgap[i][j]);
        }
    // printf ("DEBUG::=>更新完毕...\r\n");
}

void Gd_Simulate
(double ***model_gap,double ****model,
 int pre_ly_index,int cur_ly_index,
 int pre_nds,int cur_nds,int sample_nums)
{
    double ***parm = (*model);
    double **pre_2_cur_parm = parm[pre_ly_index];
    for(int i=0;i<pre_nds;i++)
        for(int j=0;j<cur_nds;j++)
        {
            pre_2_cur_parm[i][j] = G*pre_2_cur_parm[i][j] - (A*model_gap[pre_ly_index][i][j])/sample_nums;
            // if (pre_ly_index != 0)
            //     printf ("DEBUG::L<%d>K<%d>2<%d>::[%0.6lf]\r\n",pre_ly_index,i,j,pre_2_cur_parm[i][j]);
        }
}

void Clear_Model_DGap(double ****model_gap,int layers,unsigned int *layer_nodes)
{
    double ***dgap = (*model_gap);
    for (int i=0;i<layers-1;i++)                       
    for (int j=0;j<layer_nodes[i];j++)
        memset(dgap[i][j],0,sizeof(double)*layer_nodes[i+1]);
}

Mat Polling_Layer(Mat src,int ly_sz)
{
    int out_size = src.rows/ly_sz;
    Mat out(Size(out_size,out_size),CV_8UC1,Scalar(0));
    // printf ("FINAL_SIZE::[%d]\r\n",out_size);
    for (int y=0;y<src.rows;y+=ly_sz)
    {
        for (int x=0;x<src.cols;x+=ly_sz)
        {
            unsigned char tmps = 0;
            for (int ystep=0;ystep<ly_sz;ystep++)
            for (int xstep=0;xstep<ly_sz;xstep++)
                tmps = MAX(tmps,src.at<uchar>(Point(x+xstep,y+ystep)));
            out.at<uchar>(Point(x/ly_sz,y/ly_sz))=tmps;
        }
    }
    return out;
}

void Test_Model(const char *file_route,unsigned int *layer_nodes,
unsigned int layers,double ***layer,double ***model,char** class_names,
unsigned int ct)
{
    double **ly = (*layer);
    unsigned char *ts_tensor;
    double *data_set = (double*)malloc(sizeof(double)*Final_Size*Final_Size*Fliter_Nums);
    Mat ts_tmps = imread(file_route);
    if (!ts_tmps.empty())
    {
        resize(ts_tmps,ts_tmps,Size(IMG_WIDTH,IMG_HEIGHT),0,0,INTER_LINEAR);
        cvtColor(ts_tmps,ts_tmps,CV_RGB2GRAY);
        Mat newtmps[Fliter_Nums];
        newtmps[0] = Filiter(ts_tmps,3,2,PREWITT_HORI,IMG_HEIGHT);
        newtmps[1] = Filiter(ts_tmps,3,2,PREWITT_VERT,IMG_HEIGHT);
        newtmps[2] = Filiter(ts_tmps,3,2,SCHARR_HORI,IMG_HEIGHT);
        newtmps[3] = Filiter(ts_tmps,3,2,SCHARR_VERT,IMG_HEIGHT);
        newtmps[4] = Filiter(ts_tmps,3,2,SOBEL_HORI,IMG_HEIGHT);
        newtmps[5] = Filiter(ts_tmps,3,2,SOBEL_VERT,IMG_HEIGHT);
        newtmps[0] = Polling_Layer(newtmps[0],6);
        newtmps[1] = Polling_Layer(newtmps[1],6);
        newtmps[2] = Polling_Layer(newtmps[2],6);
        newtmps[3] = Polling_Layer(newtmps[3],6);
        newtmps[4] = Polling_Layer(newtmps[4],6);
        newtmps[5] = Polling_Layer(newtmps[5],6);   
        //--图片数据::[Final_Size^2x1]--
        //--展开图片特征数据--
        ts_tensor = Extend_Tensor(newtmps,Fliter_Nums);
    }
    for (int j=0;j<Final_Size*Final_Size*Fliter_Nums;j++)
        data_set[j] = ts_tensor[j] / 255.0;
    for (int i=0;i<layer_nodes[0];i++)
            ly[0][i] = data_set[i];
    //--正向传播--
    for (int i=1;i<layers;i++)
    {
        Set_Simulate(model,i-1,i,
        &ly,layer_nodes[i-1],layer_nodes[i],layers);
    }
    //--预测结果--
    for (int i=0;i<ct;i++)
        printf ("第<%d>张图片的预测结果::=> \
        <%s>::probility::[%0.3lf]\r\n\n",0,class_names[i],ly[layers-1][i]);
}

void Rate_Learn(unsigned int *layer_nodes,unsigned int tensor_count,double **data_set,
 double **layer,unsigned int layers,double ***nodes_parm,char** class_names,
 unsigned int N,int *labels,unsigned int ct)
{
    //--fit--
    //--随机取N图片张量--
    int correct = 0;
    for (int j=0;j<N;j++)
    {
        int pice = random()%tensor_count;
        for (int i=0;i<layer_nodes[0];i++)
            layer[0][i] = data_set[pice][i];
        //--正向传播--
        for (int i=1;i<layers;i++)
        {
            Set_Simulate(nodes_parm,i-1,i,
            &layer,layer_nodes[i-1],layer_nodes[i],layers);
        }
        //--预测结果--
        for (int i=0;i<ct;i++)
            printf ("第<%d>张图片的预测结果::=> \
            <%s>::probility::[%0.3lf]\r\n",pice,class_names[i],layer[layers-1][i]);
        printf ("\r\n");
        int index = -1;
        for (int i=0;i<ct;i++)
        {
            double tmps = 0;
            if (tmps < layer[layers-1][i])
            {
                tmps = layer[layers-1][i];
                index = i;
            }
        }
        if (index == labels[pice])
            correct++;
    }
    printf ("Correct Rate ::=> [%0.5lf]\r\n",correct/1.0/N);
}

