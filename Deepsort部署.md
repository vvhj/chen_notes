# Deepsort

下面讲的是如何讲deepsort部署到华为盒子上

1. 克隆仓库

   `git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git`
   
2. 安装依赖

   `pip install -r requirements.txt`

## 1. 模型转换

deepsort有一个特征提取模块，下载参数权重ckpt.t7

`https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6`

由于yolo模型检测到的目标数不是固定的，对ckpt.t7来说要支持动态批次；以下命令支持1～32的动态批次

`atc --model=./ckpt.onnx --framework=5 --output=ckpt  --input_shape="images:-1,3,128,64" --soc_version=Ascend310 --dynamic_batch_size="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"`

最终，得到ckpt.om

## 2. 运行流程

由于要支持动态批次，200DK上的代码要做相应修改。访问网站：

`https://support.huaweicloud.com/Atlas200DK202/`

![deepsort1](/home/chen/Desktop/文档/deepsort/deepsort1.png)

根据文档：应用开发（Python）/接口调用流程/模型推理/基本的模型推理流程，可知：

`模型执行前要设置动态batch`

![deepsort2](/home/chen/Desktop/文档/deepsort/deepsort2.png)

再根据文档：应用开发（Ptyhon）/接口调用流程/模型推理/设置动态Batch，可知分两步：

1. 根据名称获取动态Batch的index：acl.mdl.get_input_by_name
2. 设置动态Batch：acl.mdl.set_dynamic_batch_size

![deepsort3](/home/chen/Desktop/文档/deepsort/deepsort3.png)但这样还是不够的，再根据文档：应用开发（Ptyhon)/接口调用流程/模型推理/准备模型推理的输入/输出数据；可知在执行模型之前，我们需要先申请输入/输出内存，更加重要的是我们需要申请的是动态内存。在关键说明部分提到，如果模型涉及动态Batch，建议调用函数：`get_input_size_by_index`接口获取，且该接口是最大档位内存，确保内存够用。

## 3. 代码修改

### 3.1 文件结构

chen_om工程文件结构如下：

![6](/home/chen/Desktop/文档/deepsort/6.png)

ydeepsort结构如下：

![8](/home/chen/Desktop/文档/deepsort/8.png)

### 3.2 atlas_utils

下面开始修改om支持模型框架atlas_utils的代码，主要是保证ckpt.om模型在200dk上能顺利运行。打开文件chen_om/atlas_utils/acl_model.py，定位到execute方法，在`ret = acl.mdl.execute`代码前添加如下代码：

```python
if model == 'deepsort':
    # 获取index
	self.index, ret = acl.mdl.get_input_index_by_name(self._model_desc, 'ascend_mbatch_shape_data')
    # 检查一下index是否正确
	utils.check_ret("index", ret)
    # 设置动态批次
	acl.mdl.set_dynamic_batch_size(self._model_id, self._input_dataset, self.index, input_list[0].shape[0])
```

申请内存，定位到代码./acl_model.py，_parse_input_data方法：

```python
...
elif isinstance(input_data, np.ndarray):
    ptr = acl.util.numpy_to_ptr(input_data)
    # 申请32档位所需要内存的大小
    size = acl.mdl.get_input_size_by_index(self._model_desc, index)
    # 调用申请内存的函数
    data = self._copy_input_to_device(ptr, size, index)
    
```

修改`_copy_input_to_device`方法，之前代码默认当前size等于缓存的size时候内存是不重新申请的；在我们的应用场景下，先是调用yolo4.om，所以在执行ckpt.om之前先要申请我们需要的内存： 

```python
 ret = acl.rt.memcpy(buffer_item['addr'], size,
                                input_ptr, size,
                                const.ACL_MEMCPY_DEVICE_TO_DEVICE)
```

创建inputdataset：

```python
dataset_buffer = acl.create_data_buffer(data, size)
 _, ret = acl.mdl.add_dataset_buffer(self._input_dataset, dataset_buffer)
```

同样，创建outputdataset：

```python
# 获取32档位所需要的输出内存
size = acl.mdl.get_output_size_by_index(self._model_desc, i)
buf, ret = acl.rt.malloc(size, const.ACL_MEM_MALLOC_NORMAL_ONLY)
# 创建outputdataset
dataset_buffer = acl.create_data_buffer(buf, size)
_, ret = acl.mdl.add_dataset_buffer(dataset, dataset_buffer)
```

## 4. deepsort代码

所有代码分为3部分：

1. atlas_utils om模块
2. ydeepsort yolov4及deepsort支持模块
3. detect_rtsp_deepsort.py 核心运行代码



打开`detect_rtsp_deepsort.py`文件定位到注释`do deepsort`，此处det已经是yolov4模型在检测区域内检测出来的bboxes。首先把xyxy坐标从相对尺寸换成像素尺寸：

```python
det[:, [0, 2]] = (det[:, [0, 2]] * MODEL_WIDTH * x_scale).round()
det[:, [1, 3]] = (det[:, [1, 3]] * MODEL_HEIGHT * y_scale).round()
```

xyxy转换到xywh坐标：

```python
xywhs = xyxy2xywh(det[:, 0:4])
# 置信度
confs = det[:, 5]
# 类别
clss = det[:, 4]
```

根据预测xywh，从原图中截取目标位置图像：

```python
for box in xywhs:
    # 原图下的坐标
	x1, y1, x2, y2 = _xywh_to_xyxy(box, height, width)
	# 截取并添加到集合
	im = im0s[y1:y2, x1:x2]
	if (im.shape[0] != 0) and (im.shape[1] != 0):
       print("im:", im.shape)
       im_crops.append(im)
```

用上一步的ckpt.om抽取特征：

```python
if im_crops:
    # 把目标位置图像都resize要统一大小
    im_batch = _preprocess(im_crops)
    print("im_batch:", im_batch.shape)
    # 在act工具下，经过动态批次转换的.om，都会有两个输入；第一个是图像，第二个则是输入批次形状且必须为np.array类型
    features = model_extractor.execute([im_batch, np.array(im_batch.shape)], 'deepsort')
    # 由于我们输出内存大小定义为32批次，所以我们需要根据输入的batch大小，截取我们所需要的特征
    features = features[0][0:im_batch.shape[0], :]
```

更具特征我们可以开始作跟踪了：

```python
# 转换到top-left wh
bbox_tlwh = _xywh_to_tlwh(xywhs)
# 获得detections信息
detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
                                    confs) if conf > MIN_CONFIDENCE]
```

更新tracker：

```python
tracker.predict()
tracker.update(detections, clss)
```

获取输出：

```python
outputs = []
# 迭代.tracks
for track in tracker.tracks:
    if not track.is_confirmed() or track.time_since_update > 1:
        continue
    # box
    box = track.to_tlwh()
    x1, y1, x2, y2 = _tlwh_to_xyxy(box, height, width)
    # id
    track_id = track.track_id
    class_id = track.class_id
    # 坐标信息，跟踪id，类别id
    outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
```

后续就是上传到平台的后处理了

## 5. 其他

用rtsp取流，在高速场景下，yolov4全类别检测，车辆跟踪可以达到4～5帧；然后，在靠近摄像头车辆运行比较缓慢的情况下跟踪效果比较好，由于帧率比较低，快速的车辆是无法跟踪到的；还有，测试了10个异常停车视频，效果还是很不错的。
