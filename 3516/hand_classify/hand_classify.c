/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * 该文件提供了基于yolov2的手部检测以及基于resnet18的手势识别，属于两个wk串行推理。
 * 该文件提供了手部检测和手势识别的模型加载、模型卸载、模型推理以及AI flag业务处理的API接口。
 * 若一帧图像中出现多个手，我们通过算法将最大手作为目标手送分类网进行推理，
 * 并将目标手标记为绿色，其他手标记为红色。
 *
 * This file provides hand detection based on yolov2 and gesture recognition based on resnet18,
 * which belongs to two wk serial inferences. This file provides API interfaces for model loading,
 * model unloading, model reasoning, and AI flag business processing for hand detection
 * and gesture recognition. If there are multiple hands in one frame of image,
 * we use the algorithm to use the largest hand as the target hand for inference,
 * and mark the target hand as green and the other hands as red.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "sample_comm_nnie.h"hand_classify/hand_gesture.wk
#include "sample_media_ai.h"
#include "ai_infer_process.h"
#include "yolov2_hand_detect.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "misc_util.h"
#include "hisignalling.h"
#include <termios.h>

#include <math.h>

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define HAND_FRM_WIDTH     640
#define HAND_FRM_HEIGHT    384
#define DETECT_OBJ_MAX     32
#define RET_NUM_MAX        4
#define DRAW_RETC_THICK    2    // Draw the width of the line
#define WIDTH_LIMIT        32
#define HEIGHT_LIMIT       32
#define IMAGE_WIDTH        224  // The resolution of the model IMAGE sent to the classification is 224*224
#define IMAGE_HEIGHT       224
#define MODEL_FILE_GESTURE    "/userdata/models/pest_detect/pest_detect_inst.wk" // darknet framework wk model
// #define MODEL_FILE_GESTURE    "/userdata/models/hand_classify/hand_gesture.wk"
#define max 85
#define IS_MAOMAOCHONG(num) \
    (num == 0u || num == 1u || num == 3u || num == 4u || num == 14u || \
     num == 17u || num == 18u || num == 19u || num == 20u || num == 22u || \
     num == 23u || num == 28u || num == 36u || num == 40u || num == 41u || \
     num == 46u || num == 47u || num == 83u)  //类别信息，这里所有类别都默认为毛毛虫
static int biggestBoxIndex;
static IVE_IMAGE_S img;
static DetectObjInfo objs[DETECT_OBJ_MAX] = {0};
static RectBox boxs[DETECT_OBJ_MAX] = {0};
static RectBox objBoxs[DETECT_OBJ_MAX] = {0};
static RectBox remainingBoxs[DETECT_OBJ_MAX] = {0};
static RectBox cnnBoxs[DETECT_OBJ_MAX] = {0}; // Store the results of the classification network
static RecogNumInfo numInfo[RET_NUM_MAX] = {0};
static IVE_IMAGE_S imgIn;
static IVE_IMAGE_S imgDst;
static VIDEO_FRAME_INFO_S frmIn;
static VIDEO_FRAME_INFO_S frmDst;
int uartFd = 0;

unsigned int TO3861MsgSend(int fd, char *buf, unsigned int dataLen);

// 定义相机内参和畸变系数结构体
typedef struct {
    double fx, fy;   // 焦距
    double cx, cy;   // 光心坐标
} CameraIntrinsics;

typedef struct {
    double k1, k2;   // 径向畸变系数
    double p1, p2;   // 切向畸变系数
} DistortionCoefficients;

// 全局变量，存储相机参数和畸变系数
CameraIntrinsics intrinsics = { 130.1650538, 121.5755856, 399.975092, 239.991694 };
DistortionCoefficients distortion = { 0.011590, -0.000140, 0.005386, 0.000301 };

/*
 * 加载手部检测和手势分类模型
 * Load hand detect and classify model
 */
HI_S32 Yolo2HandDetectResnetClassifyLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = CnnCreate(&self, MODEL_FILE_GESTURE);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    HandDetectInit(); // Initialize the hand detection model
    //SAMPLE_PRT("Load hand detect claasify model success\n");
    /*
     * Uart串口初始化
     * Uart open init
     */
    uartFd = UartOpenInit();
    if (uartFd < 0) {
        printf("uart1 open failed\r\n");
    } else {
        printf("uart1 open successed\r\n");
    }
    return ret;
}

/*
 * 卸载手部检测和手势分类模型
 * Unload hand detect and classify model
 */
HI_S32 Yolo2HandDetectResnetClassifyUnload(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    HandDetectExit(); // Uninitialize the hand detection model
    close(uartFd);
    //SAMPLE_PRT("Unload hand detect claasify model success\n");

    return 0;
}

/*
 * 获得最大的手
 * Get the maximum hand
 */
static HI_S32 GetBiggestHandIndex(RectBox boxs[], int detectNum)
{
    HI_S32 handIndex = 0;
    HI_S32 biggestBoxIndex = handIndex;
    HI_S32 biggestBoxWidth = boxs[handIndex].xmax - boxs[handIndex].xmin + 1;
    HI_S32 biggestBoxHeight = boxs[handIndex].ymax - boxs[handIndex].ymin + 1;
    HI_S32 biggestBoxArea = biggestBoxWidth * biggestBoxHeight;

    for (handIndex = 1; handIndex < detectNum; handIndex++) {
        HI_S32 boxWidth = boxs[handIndex].xmax - boxs[handIndex].xmin + 1;
        HI_S32 boxHeight = boxs[handIndex].ymax - boxs[handIndex].ymin + 1;
        HI_S32 boxArea = boxWidth * boxHeight;
        if (biggestBoxArea < boxArea) {
            biggestBoxArea = boxArea;
            biggestBoxIndex = handIndex;
        }
        biggestBoxWidth = boxs[biggestBoxIndex].xmax - boxs[biggestBoxIndex].xmin + 1;
        biggestBoxHeight = boxs[biggestBoxIndex].ymax - boxs[biggestBoxIndex].ymin + 1;
    }

    if ((biggestBoxWidth == 1) || (biggestBoxHeight == 1) || (detectNum == 0)) {
        biggestBoxIndex = -1;
    }

    return biggestBoxIndex;
}


//通信函数:将识别到害虫的数量、类别和距离信息通过此函数传递给3861
/*三个约束：距离约束：通过测距函数测出所有检测框中心点到相机距离，遍历后选出距离相机最小的距离，将它与我们设定的喷枪射程（0.4米）作对比，如小于，则将通信内容第2位设为1，代表喷药，否则设为0，代表不喷
           数量约束：通过检测目标框个数，将喷枪档位分成小(1)、中(2)、大(3)三档，分别对应通信内容第3位
           类别约束：通过检测虫灾类别，如为毛毛虫，则将通信内容第5位设为1，代表常规喷枪位
                                     如为其他虫，则将通信内容第5位设为2，代表辅助喷枪位
            举例：开启喷枪，中档，常规喷枪，则通讯内容为0xaa 0x55 0x1 0x6 0 0x1 0xff */
static void PestDetect_Num_Species_Distance_Flag(double minDistance, const RecogNumInfo resBuf, int objNum)
{
    unsigned char SendBuffer[4];
    int PestNum = objNum;

    if (minDistance <= 0.4) {
        SendBuffer[0] = 1;
    } else {
        SendBuffer[0] = 0;
    }
    
    HI_CHAR *PestSpecies = NULL;

    if (PestNum <= 1) {
        if (IS_MAOMAOCHONG(resBuf.num)) {
            PestSpecies = "maomaochong";
            SendBuffer[1] = 1; SendBuffer[2] = 0; SendBuffer[3] = 1;
            TO3861MsgSend(uartFd, SendBuffer, sizeof(SendBuffer));
            printf("send Num:Little beng:2\r\n"); 
            SAMPLE_PRT("----PestSpecies----:%s\n", PestSpecies);
        } else {
            PestSpecies = "qitachong!";
            SendBuffer[1] = 1; SendBuffer[2] = 0; SendBuffer[3] = 2;
            TO3861MsgSend(uartFd, SendBuffer, sizeof(SendBuffer));
            printf("send Num:Little beng:1\r\n"); 
            SAMPLE_PRT("----PestSpecies name----:%s\n", PestSpecies);
        }
    } 
    else if (PestNum <= 3) {
        if (IS_MAOMAOCHONG(resBuf.num)) {
            PestSpecies = "maomaochong";
            SendBuffer[1] = 2; SendBuffer[2] = 0; SendBuffer[3] = 1;
            TO3861MsgSend(uartFd, SendBuffer, sizeof(SendBuffer));
            printf("send Num:Medium beng:2\r\n"); 
            SAMPLE_PRT("----PestSpecies----:%s\n", PestSpecies);
        } else {
            PestSpecies = "qitachong!";
            SendBuffer[1] = 2; SendBuffer[2] = 0; SendBuffer[3] = 2;
            TO3861MsgSend(uartFd, SendBuffer, sizeof(SendBuffer));
            printf("send Num:Medium beng:1\r\n"); 
            SAMPLE_PRT("----PestSpecies name----:%s\n", PestSpecies);
        }
    }
    else {
        if (IS_MAOMAOCHONG(resBuf.num)) {
            PestSpecies = "maomaochong";
            SendBuffer[1] = 3; SendBuffer[2] = 0; SendBuffer[3] = 1;
            TO3861MsgSend(uartFd, SendBuffer, sizeof(SendBuffer));
            printf("send Num:Large beng:2\r\n"); 
            SAMPLE_PRT("----PestSpecies----:%s\n", PestSpecies);
        } else {
            PestSpecies = "qitachong!";
            SendBuffer[1] = 3; SendBuffer[2] = 0; SendBuffer[3] = 2;
            TO3861MsgSend(uartFd, SendBuffer, sizeof(SendBuffer));
            printf("send Num:Large beng:1\r\n"); 
            SAMPLE_PRT("----PestSpecies name----:%s\n", PestSpecies);
        }
    }
}

int UartSend_1(int fd, char *buf, int len)
{
    int ret = 0;
    int count = 0;
    char *sendBuf = buf;
    int sendLen = len;

    tcflush(fd, TCIFLUSH);

    while (sendLen > 0) {
        ret = write(fd, (char*)sendBuf + count, sendLen);
        if (ret < 1) {
            printf("write data below 1 byte % d\r\n", ret);
            break;
        }
        count += ret;
        sendLen -= ret;
    }

    return count;
}

unsigned int MakeDataPackage(HisignallingProtocalType *buf,
    unsigned int len, unsigned char *hisignallingDataBuf)
{
    unsigned int packageLen = 0;
    unsigned int DataPackLen = len;

    (void)memcpy(hisignallingDataBuf, buf->frameHeader, HISGNALLING_MSG_FRAME_HEADER_LEN);
    (void)memcpy(&hisignallingDataBuf[HISGNALLING_MSG_FRAME_HEADER_LEN],
        buf->hisignallingMsgBuf, DataPackLen);
    (void)memcpy(&hisignallingDataBuf[HISGNALLING_MSG_FRAME_HEADER_LEN + DataPackLen],
        &(buf->endOfFrame), HISIGNALLING_MSG_HEADER_LEN);

    packageLen = HISGNALLING_MSG_FRAME_HEADER_LEN + DataPackLen + HISIGNALLING_MSG_HEADER_LEN;
    
    return packageLen;
}

unsigned int TO3861MsgSend(int fd, char *buf, unsigned int dataLen)
{
    unsigned int ret = 0;
    HisignallingProtocalType hisignallingMsg = {0};
    unsigned char hisignallingSendBuf[HISIGNALLING_MSG_BUFF_LEN] = {0};
    unsigned int hisignallingPackageLen = 0;

    hisignallingMsg.frameHeader[0]= 0xAA; /* Protocol head data 1 */
    hisignallingMsg.frameHeader[1]= 0x55; /* Protocol head data 2 */
    (void)memcpy(hisignallingMsg.hisignallingMsgBuf, buf, dataLen);
    hisignallingMsg.endOfFrame = 0xFF; /* Protocol tail data */

    hisignallingPackageLen = MakeDataPackage(&hisignallingMsg, dataLen, hisignallingSendBuf);
    if (!hisignallingPackageLen) {
        printf("hisignalling_data_package failed\r\n");
        return -1;
    }
    if (*hisignallingSendBuf == 0) {
        printf("hisignalling send buf is null!\r\n");
        return -1;
    }

    ret = UartSend_1(fd, (char*)hisignallingSendBuf, hisignallingPackageLen);
    if (ret < 0) {
        printf("write data failed\r\n");
        return -1;
    }

    return 0;
}


// 去畸变函数
void undistortPoints(double x, double y, const CameraIntrinsics* intrinsics, const DistortionCoefficients* distortion, double* x_out, double* y_out) {
    
    double x_normalized = (x - intrinsics->cx) / intrinsics->fx;
    double y_normalized = (y - intrinsics->cy) / intrinsics->fy;
    
    double x_distorted = x_normalized;
    double y_distorted = y_normalized;

    for (int i = 0; i < 5; ++i) {
        double r2 = x_distorted * x_distorted + y_distorted * y_distorted;
        double radial_distortion = 1 + distortion->k1 * r2 + distortion->k2 * r2 * r2;
        double tangential_distortion_x = 2 * distortion->p1 * x_distorted * y_distorted + distortion->p2 * (r2 + 2 * x_distorted * x_distorted);
        double tangential_distortion_y = distortion->p1 * (r2 + 2 * y_distorted * y_distorted) + 2 * distortion->p2 * x_distorted * y_distorted;

        x_distorted = (x_normalized - tangential_distortion_x) / radial_distortion;
        y_distorted = (y_normalized - tangential_distortion_y) / radial_distortion;

    }

    *x_out = x_distorted * intrinsics->fx + intrinsics->cx;
    *y_out = y_distorted * intrinsics->fy + intrinsics->cy;
}


// 计算目标框中心点深度函数
double estimateDistance(double x_center, double y_center, double bbox_height, double H_real, const CameraIntrinsics* intrinsics, const DistortionCoefficients* distortion) {
    double x_undistorted, y_undistorted;

    undistortPoints(x_center, y_center, intrinsics, distortion, &x_undistorted, &y_undistorted);

    double x_pixel = x_undistorted;
    double y_pixel = y_undistorted;

    double fx = intrinsics->fx;
    double fy = intrinsics->fy;

    double Zx = (fx * H_real) / bbox_height;  
    double Zy = (fy * H_real) / bbox_height;  

    double Z = (Zx + Zy) / 2.0;

    return Z;
}

HI_S32 Yolo2HandDetectResnetClassifyCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm, int fd)
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    HI_S32 resLen = 0;
    int objNum;
    int ret;
    int num = 0;
    double minDistance = 100;  
    int minDistanceIndex = -1;

    ret = FrmToOrigImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "hand detect for YUV Frm to Img FAIL, ret=%#x\n", ret);

    objNum = HandDetectCal(&img, objs); // Send IMG to the detection net for reasoning
    for (int i = 0; i < objNum; i++) {
        cnnBoxs[i] = objs[i].box;
        RectBox *box = &objs[i].box;
        RectBoxTran(box, HAND_FRM_WIDTH, HAND_FRM_HEIGHT,
            dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
        SAMPLE_PRT("yolo2_out: {%d, %d, %d, %d}\n", box->xmin, box->ymin, box->xmax, box->ymax);
        boxs[i] = *box;

        double x_center = (box->xmin + box->xmax) / 2.0;
        double y_center = (box->ymin + box->ymax) / 2.0;

        double x_undistorted, y_undistorted;
        undistortPoints(x_center, y_center, &intrinsics, &distortion, &x_undistorted, &y_undistorted);

        
        double bbox_height = box->ymax - box->ymin;
        double H_real = 0.06;  //真实毛毛虫物体尺度
        double Z = 10 * estimateDistance(x_undistorted, y_undistorted, bbox_height, H_real, &intrinsics, &distortion);//因相机为可变焦相机，故10是经过反复实验调参得到

        //遍历得到最小距离
        if (Z < minDistance) {
            minDistance = Z;
            minDistanceIndex = i;
        }

        //打印，方便实验测试
        if (minDistanceIndex >= 0) {
        SAMPLE_PRT("Minimum Distance Z: %f\n", minDistance);
        SAMPLE_PRT("Corresponding Box: {%d, %d, %d, %d}\n", boxs[minDistanceIndex].xmin, boxs[minDistanceIndex].ymin, boxs[minDistanceIndex].xmax, boxs[minDistanceIndex].ymax);
    }
    }

    biggestBoxIndex = GetBiggestHandIndex(boxs, objNum);
    SAMPLE_PRT("biggestBoxIndex:%d, objNum:%d\n", biggestBoxIndex, objNum);

    /*
     * 当检测到对象时，在DSTFRM中绘制一个矩形
     * When an object is detected, a rectangle is drawn in the DSTFRM
     */
    if (biggestBoxIndex >= 0) {
        objBoxs[0] = boxs[biggestBoxIndex];
        MppFrmDrawRects(dstFrm, objBoxs, 1, RGB888_GREEN, DRAW_RETC_THICK); // Target hand objnum is equal to 1

        for (int j = 0; (j < objNum) && (objNum > 1); j++) {
            if (j != biggestBoxIndex) {
                remainingBoxs[num++] = boxs[j];
                /*
                 * 其他手objnum等于objnum -1
                 * Others hand objnum is equal to objnum -1
                 */
                MppFrmDrawRects(dstFrm, remainingBoxs, objNum - 1, RGB888_RED, DRAW_RETC_THICK);
            }
        }

        /*
         * 裁剪出来的图像通过预处理送分类网进行推理
         * The cropped image is preprocessed and sent to the classification network for inference
         */
        ret = ImgYuvCrop(&img, &imgIn, &cnnBoxs[biggestBoxIndex]);
        SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "ImgYuvCrop FAIL, ret=%#x\n", ret);

        if ((imgIn.u32Width >= WIDTH_LIMIT) && (imgIn.u32Height >= HEIGHT_LIMIT)) {
            COMPRESS_MODE_E enCompressMode = srcFrm->stVFrame.enCompressMode;
            ret = OrigImgToFrm(&imgIn, &frmIn);
            frmIn.stVFrame.enCompressMode = enCompressMode;
            SAMPLE_PRT("crop u32Width = %d, img.u32Height = %d\n", imgIn.u32Width, imgIn.u32Height);
            ret = MppFrmResize(&frmIn, &frmDst, IMAGE_WIDTH, IMAGE_HEIGHT);
            ret = FrmToOrigImg(&frmDst, &imgDst);
            ret = CnnCalImg(self, &imgDst, numInfo, sizeof(numInfo) / sizeof((numInfo)[0]), &resLen);
            SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "CnnCalImg FAIL, ret=%#x\n", ret);
            HI_ASSERT(resLen <= sizeof(numInfo) / sizeof(numInfo[0]));
            PestDetect_Num_Species_Distance_Flag(minDistance, numInfo[0], objNum);
            MppFrmDestroy(&frmDst);
        }
        IveImgDestroy(&imgIn);
    }

    return ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
