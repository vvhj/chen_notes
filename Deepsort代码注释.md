# Deepsort代码注释

下面是deepsort代码注释，文件是detect_rtsp_deepsort.py，定位到:

```python
# 开始跟踪的处理
if det is not None and len(det):
...
```

所有代码均在xsy_om工程中：

![m19](/home/chen/Desktop/文档/deepsort/m19.png)

## 1. 特征提取

相对尺度转换为像素尺度，最后坐标得到原图像素尺度的（左上角x, 左上角y, 右下角x, 右下角y）：

```python
# 相对坐标 * 原图像素尺度 * 缩放尺寸
det[:, [0, 2]] = (det[:, [0, 2]] * MODEL_WIDTH * x_scale).round()
det[:, [1, 3]] = (det[:, [1, 3]] * MODEL_HEIGHT * y_scale).round()
```

转化为(中心x，中心y，宽度w，长度h)：

```python
xywhs = xyxy2xywh(det[:, 0:4]) # xywh 
confs = det[:, 5] # 置信度
clss = det[:, 4] # 类别
```

从原图im0s中截取目标区域：

```python
height, width = orig_shape # 原图高，宽
im_crops = [] # list存储目标区域
for box in xywhs:
    x1, y1, x2, y2 = _xywh_to_xyxy(box, height, width) # 转换为x1y1x2y2坐标
    im = im0s[y1:y2, x1:x2] # 截取图片
    if (im.shape[0] != 0) and (im.shape[1] != 0):
       print("im:", im.shape)
       im_crops.append(im) # 添加进队列
```

特征提取：

```python
if im_crops:
		# 把目标区域统一resize成（64，128）大小
        im_batch = _preprocess(im_crops) 
        # 网络结构为：
        # 3 x 128 x 64  卷基层
        # 32 x 64 x 32  卷基层
        # 32 x 64 x 32  卷积层
        # 64 x 32 x 16  卷积层
        # 128 x 16 x 8	卷积层
        # 256 x 8 x 4   平均池化层
        # 256 x 1 x 1	全连接层
        features = model_extractor.execute([im_batch, np.array(im_batch.shape)], 'deepsort')
        # 截取所需部分特征
        features = features[0][0:im_batch.shape[0], :]
else:
        features = np.array([])
```

###  1.1 _preprocess

```python
size = (64, 128)

# 标准化
def norm(im):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # output = im.copy()
    for i in np.arange(0, 3):
        im[:, :, i] = (im[:, :, i] - mean[i]) / std[i]
    return im

def _preprocess(im_crops):
    """
    TODO:
        1. to float with scale from 0 to 1
        2. resize to (64, 128) as Market1501 dataset did
        3. concatenate to a numpy array
        3. to torch Tensor
        4. normalize
    """
	# 目标区域统一resize成（64，128）并归一化
    def _resize(im, size):
        return cv2.resize(im.astype(np.float32) / 255., size)

    # 深度维度放在前，然后扩展一个维度，np.concatenate连接起来，最终得到形状（批次大小，深度，128，64）
    im_batch = np.concatenate([np.expand_dims(norm(_resize(im, size)).transpose(2, 0, 1), axis=0) for im in im_crops], axis=0)
   
    return im_batch
```

## 2. Detection类

先来认识类Detection，代表了在一张图片中的一个预测框：

```python
class Detection(object):
        def __init__(self, tlwh, confidence, feature):
        	self.tlwh = np.asarray(tlwh, dtype=np.float) # 像素坐标（左上角x，左上角y，宽度，高度）
        	self.confidence = float(confidence) # 置信度
        	self.feature = np.asarray(feature, dtype=np.float32) # 对应的512维向量
        
        # 转换为（左上角x，左上角y，右下角x，右下角y）
        def to_tlbr(self):
        	ret = self.tlwh.copy()
        	ret[2:] += ret[:2]
        	return ret
        
        # 转换为（中心x，中心y，长宽比，高），其中长宽比=width/height
    	def to_xyah(self):
        	ret = self.tlwh.copy()
        	ret[:2] += ret[2:] / 2
        	ret[2] /= ret[3]
        	return ret
```

然后看如下代码：

```python
bbox_tlwh = _xywh_to_tlwh(xywhs) # 转化为（左上角x，左上角y，宽，高）
# 根据置信度过滤掉一部分box，生成Detection类list
detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
                                    confs) if conf > MIN_CONFIDENCE]
```

## 3. 更新跟踪器

首先看跟踪器的初始化：

```python
max_cosine_distance = MAX_DIST	# 最大cosine距离
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, NN_BUDGET)	# 距离度量
tracker = Tracker(metric, max_iou_distance=max_cosine_distance, max_age=MAX_AGE, n_init=N_INIT)
```

### 3.1 最临近距离度量

NearestNeighborDistanceMetric，一个最近邻居距离度量：对每一个目标，返回和当前观测到的任意样本的最近距离。样本通过一个key为id的字典存起来。

1. cosine距离
2. 大于最大cosine距离被认为是无效匹配
3. self.budget最大样本数量
4. self.samples = {} 字典，目标id -> 当前以观测到的所有样本

```python
# max_cosine_distance = 0.2
# NN_BUDGET = 100
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, NN_BUDGET)

# 最近临近距离度量：对每一个目标，返回观测到目前为止任意样本的最近距离
class NearestNeighborDistanceMetric(object):
    # 初始化
    def __init__(self, metric, matching_threshold, budget=None):
        
        # 使用cosine距离 or 欧式距离
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
            
        # 匹配阈值：大于这个距离被认为是无效匹配
        self.matching_threshold = matching_threshold
        self.budget = budget
        # 字典：从目标id映射到样本列表
        self.samples = {}
        
	 # 使用新的数据更新距离度
     def partial_fit(self, features, targets, active_targets)：
        # features: 一个NxM的矩阵
    	# targets: 关联目标id的list
     	# active_targets: 出现在当前场景中的目标
    	 for feature, target in zip(features, targets):
			self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
         self.samples = {k: self.samples[k] for k in active_targets}
    
    # 计算特征和目标的距离
    def distance(self, features, targets):
        # (目标数，特征数)
        cost_matrix = np.zeros((len(targets), len(features)))
        # 迭代
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
```



#### 3.1.1 partial_filt

```python
def partial_fit(self, features, targets, active_targets):
	# features N个特征
	# targets 目标id
	# active_targets 当前出现在场景中的目标id
    
    # 1. 后面就是往samples字典中对应的id添加feature
    # 2. 如果超过最大样本数budget就去掉一些样本
    # 3. 最后只保留当前场景活目标的样本
```

####  3.1.2 distance

计算特征和目标的距离用于计算每个特征和目标的cosine距离

```
def distance(self, features, targets):
	# features N个M维特征
	# targets 当前场景中的目标id
	
	# 计算距离矩阵
```

返回一个[len(targets), len(features)]形状的矩阵，其中[i, j]代表targets[i]和features[j]的距离。

### 3.2 Tracker

然后看如下代码，整个过程是通过卡尔曼滤波器先预测k时刻的状态（先验分布），然后结合观测（似然）跟新k时刻的状态（后验分布），其实就是个贝叶斯原理的应用：

```python
tracker.predict()
tracker.update(detections, clss)
```

首先来认识一下多目标跟踪器Tracker类：

```python
class Tracker:
	# 初始化
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        # self.metric = metric cosine距离测量
        # self.max_iou_distance = max_iou_distance 最大iou距离
        # self.max_age = max_age 最大存在70帧
        # self.n_init = n_init 一个tracker在初始化阶段的帧数
        # self.kf = kalman_filter.KalmanFilter() 卡尔曼滤波器
        # self.tracks = [] List[Track]当前活动的追踪器
        # self._next_id = 1
     
     # 对当前活动的Track进行卡尔曼滤波
     def predict(self):
        for track in self.tracks:
            track.predict(self.kf)
```


####  3.2.1 predict

没什么好讲的，就是迭代当前存活的Track类，然后调用相应的KalmanFilter的predict函数。

#### 3.2.2 update

执行测量跟新（似然）以及跟踪管理

```python
def update(self, detections, classes):
    # detections: 当前的观测信息，也就是yolov4的预测框信息
    
    # 运行匹配小瀑布？？ 应该翻译成匹配流程
	matches, unmatched_tracks, unmatched_detections = self._match(detections)
```

####  3.2.3 _match

```python
def _match(self, detections):
	...
	
    # 把tracks集合分为已确认和未确认的tracks
    # 代码略...
    
    # 1. 使用外表特征关联已经被确认的tracks
    # 关键是，linear_assignment.matching_cascade这个方法
	matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
     
    # 2. 使用iou关联
    iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
	unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
    # iou < 0.3
 	matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
     
    # 更新距离度量
	matches = matches_a + matches_b
    # a +b
	unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
    
    return matches, unmatched_tracks, unmatched_detections
```

####  3.2.4 gate_metric

重要！用于计算匹配的代价矩阵函数

```python
def gated_metric(tracks, dets, track_indices, detection_indices):
    features = np.array([dets[i].feature for i in detection_indices])
    targets = np.array([tracks[i].track_id for i in track_indices])
    # 3.1.2的内容，cosine距离
	cost_matrix = self.metric.distance(features, targets)
    # 3.5.3的内容
    cost_matrix = linear_assignment.gate_cost_matrix(
          self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)
	return cost_matrix
```

### 3.3 KalmanFilter

用于追踪边界框，8维的状态空间：

`x, y, a, h, vx, vy, va, vh`

1. (x, y)边界框的中心
2. a 长宽比
3. h 高度
4. 以及他们各自的速度

目标运动遵守匀速运动模型；边界框信息来自于线性观测模型的直接观测。

```python
def __init__(self):
	ndim, dt = 4, 1.
    
    # _motion_mat:运动波模型预测矩阵 [8, 8]
    # 用于更新下一步的状态
    # 为什么是这样设置？
    self._motion_mat = np.eye(2 * ndim, 2 * ndim) 
    for i in range(ndim):
        self._motion_mat[i, ndim + i] = dt
        
    self._update_mat = np.eye(ndim, 2 * ndim)
        
    # 运动模型的噪声参数，不必细究
	self._std_weight_position = 1. / 20
    self._std_weight_velocity = 1. / 160
```

#### 3.3.1 predict

卡尔曼滤波的预测步骤

```python
def predict(self, mean, covariance):
	# mean k-1时刻状态分布的均值
    # convariance k-1时刻状态分布的协方差矩阵
    
    std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
    std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
    # 1. q. np.r_ 将两个矩阵按行拼接在一起
    # 2. np.square 开平方
    # 3. np.diag 输入为一维数组，输出一个以一维数组为对角线元素的矩阵
    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
    
    # 1. mean 根据预测矩阵预测k时刻状态
    # 2. covariance 计算k时刻协方差
    # 3. motion_cov为噪声，不必细究
    mean = np.dot(self._motion_mat, mean)
    covariance = np.linalg.multi_dot((
    				self._motion_mat, covariance, self._motion_mat.T)) + motion_cov   
```

返回，预测的状态向量和协方差矩阵；未观测速度初始化未0。

####  3.3.2 预测矩阵更新状态

self._motion_mat为预测矩阵

<img src="/home/chen/Desktop/文档/deepsort/9.jpg" alt="9" style="zoom: 33%;" />

可以看出，此处运动模型假设被跟踪目标是匀速直线运动的。

#### 3.3.3 gating_distance

计算阀门距离

```python
def gating_distance(self, mean, covariance, measurements, only_position=False):
	# 从状态空间投射到测量空间
    # 其中 self._update_mat [4, 8]
    # 无非就是保留了状态的前4个
    mean, covariance = self.project(mean, covariance)
    
    # 检查矩阵是否可逆，不可逆报错
    cholesky_factor = np.linalg.cholesky(covariance)
	d = measurements - mean
    # 假设a是三角矩阵，则对x求解方程ax = b
    z = scipy.linalg.solve_triangular(
        cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
    
    # 马氏距离
	squared_maha = np.sum(z * z, axis=0)
    return squared_maha
```

返回长度为N的数组，第i个元素返回(mean, covariance) 和 measurements[i]马氏距离

### 3.4 Track类

单一目标的跟踪器

```python
def __init__(self, mean, convariance, track_id, class_id, n_init, max_age, feature=None):
    # 1. mean 初始状态分布的均值
    # 2. convariance 初始状态分布的协方差
    # 3. track_id 独一无二的id int类型
    # 4. hits 总测量更新次数
    # 5. age 跟踪的总帧数
    # 6. time_since_update 自从上次跟新的总次数
    # 7. state 当前的跟踪状态；试用=1，确认=2，删除=3 
    # 8. feature 特征向量
    # 9. n_init 从试用状态转为确认状态的连续检测的数量
    # 10. max_age 从确认状态到删除状态的最大连续跟丢的数量
    
```

#### 3.4.1 predict

1. 通过k-1时刻的状态分布，也就是均值和协方差矩阵预测k时刻的状态分布。
2. 并且跟新存活时间age，跟新次数。
3. 这个predict函数被Tracker的predict函数调用。

###  3.5 linear_assignment

线性分配策略

#### 3.5.1 matching_cascade

执行匹配流程

```python
def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    # 1.distance_metric 返回一个 NxM(N个tack，M个detcion)代价矩阵；
    #                   其中元素(i,j)代表第i个track和第j个detction关联花费
    # 2.max_distance 关联花费超过这个值就被忽略
    # 3.cascade_depth cascade深度？？
    
    # 4.tracks 已经预测k时刻的Track
    # 5.detections k时刻的观测
    
    # 6.track_indices 跟踪索引列表
    # 7.detection_indices 检测索引列表
    
    # 0~N 和 0~M
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    # 迭代深度
    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        # 没有观测还匹配什么
        if len(unmatched_detections) == 0:  
            break
            
        # 特定level只跟新特定level的Track,
        # 那就只剩下一部分索引了
		track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue
            
        # 正式开始匹配，关键是min_cost_matching方法
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l
    
    # 未匹配到的track
	unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections
    
```

####  3.5.2 min_cost_matching

解决线性匹配问题

```python
def min_cost_matching(
        distance_metric, 
         max_distance, tracks, detections, 
        track_indices=None, detection_indices=None):
    # 参数含义全部同上
    
    # dection或者track的索引为0，则返回空列表
    
    # 计算代价矩阵
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    
```

返回三个list：

1. 匹配到的track和detection的索引
2. 未匹配到的track索引
3. 未匹配到的detection索引

####  3.5.3 gate_cost_matrix

基于从Kalman filtering滤波获取的状态分布，使不可行的入口无效

```python
def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    # kf 卡尔曼滤波器
    # cost_matrix 同上
    # tracks List[track.Track]
    # detections List[detection.Detection]
    # track_indices 索引
    # detection_indices 索引
    # gated_cost Optional[float] 默认一个非常大的值
    # only_position 如果为真，在gating时只有状态分布的x,y坐标被考虑
    
    # 阀门维度
    gating_dim = 4
    # 阀门阈值 4：9.4877
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # 观测转为xyah状态
	measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    # 迭代
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        # 阀门距离
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        # 大于阈值的给一个很大的值
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
	return cost_matrix
```



##  4. 输出

tracker跟新完后，无非就是迭代tracker，输出我们需要的信息

```python
outputs = []
# 迭代
for track in tracker.tracks:
	if not track.is_confirmed() or track.time_since_update > 1:
		continue
     # box转换
     box = track.to_tlwh()
     x1, y1, x2, y2 = _tlwh_to_xyxy(box, height, width)
     track_id = track.track_id
     class_id = track.class_id
     outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
```
