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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "sample_comm_nnie.h"
#include "ai_infer_process.h"
#include "sample_media_ai.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

// #define MODEL_FILE_HAND    "/userdata/models/pest_detect/pest_detect_inst.wk" // darknet framework wk model
#define MODEL_FILE_HAND    "/userdata/models/hand_classify/hand_detect.wk"
#define PIRIOD_NUM_MAX     49 // Logs are printed when the number of targets is detected
#define DETECT_OBJ_MAX     32 // detect max obj

static uintptr_t g_handModel = 0;

//加载yolov2模型
static HI_S32 Yolo2FdLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    //创建 YOLO2 模型
    ret = Yolo2Create(&self, MODEL_FILE_HAND);
    // 如果创建失败，返回 0，否则返回模型句柄
    *model = ret < 0 ? 0 : (uintptr_t)self;
    SAMPLE_PRT("Yolo2FdLoad ret:%d\n", ret);

    return ret;
}
// 初始化手部检测
HI_S32 HandDetectInit()
{
    // 加载 YOLO2 模型并保存句柄
    return Yolo2FdLoad(&g_handModel);
}
// 卸载 YOLO2 模型
static HI_S32 Yolo2FdUnload(uintptr_t model)
{
    // 销毁 YOLO2 模型
    Yolo2Destory((SAMPLE_SVP_NNIE_CFG_S*)model);
    return 0;
}
// 退出手部检测
HI_S32 HandDetectExit()
{
    // 卸载 YOLO2 模型
    return Yolo2FdUnload(g_handModel);
}

// 进行手部检测
static HI_S32 HandDetect(uintptr_t model, IVE_IMAGE_S *srcYuv, DetectObjInfo boxs[])
{
    // 将模型句柄转换为配置结构体指针
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    int objNum;
    // 使用 YOLO2 计算图像中的检测对象
    int ret = Yolo2CalImg(self, srcYuv, boxs, DETECT_OBJ_MAX, &objNum);
    if (ret < 0) {
        SAMPLE_PRT("Hand detect Yolo2CalImg FAIL, for cal FAIL, ret:%d\n", ret);
        return ret;
    }

    return objNum;
}

// 计算手部检测结果
HI_S32 HandDetectCal(IVE_IMAGE_S *srcYuv, DetectObjInfo resArr[])
{
    // 调用 HandDetect 进行检测，并返回检测结果
    int ret = HandDetect(g_handModel, srcYuv, resArr);
    return ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
