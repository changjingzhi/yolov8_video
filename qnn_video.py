import time
import numpy as np
import cv2
import os
import aidlite
import argparse

# COCO数据集的80个类别名称
coco_class = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
              'hair drier', 'toothbrush']

# 为每个类别随机分配颜色，用于绘制检测框
colors = {name: [np.random.randint(0, 255) for _ in range(3)] for i, name in enumerate(coco_class)}

# 后处理函数
def postprocess(outputs, ratio, conf_threshold=0.5, nms_threshold=0.45):
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    
    for i in range(rows):
        classes_scores = outputs[i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        
        if maxScore >= conf_threshold:
            box = [
                outputs[i][0] - (0.5 * outputs[i][2]), outputs[i][1] - (0.5 * outputs[i][3]),
                outputs[i][2], outputs[i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # NMSBoxes returns only the result (list of indices)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_threshold, nms_threshold=nms_threshold, eta=0.5)
    
    # Now we check if result_boxes is not None or empty
    if result_boxes is not None and len(result_boxes) > 0:
        result_boxes = result_boxes.flatten()  # Use flatten() instead of reshape()
    else:
        result_boxes = []

    new_bboxes = []
    new_scores = []
    new_class_ids = []
    
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        bbox = boxes[index]
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        new_bboxes.append([round(x * ratio[0]), round(y * ratio[1]), round(w * ratio[0]), round(h * ratio[1])])
        new_scores.append(scores[index])
        new_class_ids.append(class_ids[index])

    # Check if new_scores is empty, and skip drawing if empty
    if len(new_scores) == 0:
        print("No valid scores found, skipping drawing boxes.")
        return []  # Skip this frame if no valid detections
    
    # Ensure new_scores is a 2D array (column vector)
    new_scores = np.array(new_scores)  # Convert to 2D array if necessary
    if np.ndim(new_scores) == 1:
        new_scores = np.expand_dims(new_scores, axis=1)  # Convert to shape (N, 1) for proper concatenation
    
    # Ensure new_class_ids is a 2D array (column vector)
    new_class_ids = np.expand_dims(new_class_ids, axis=1)  # Convert to shape (N, 1) for proper concatenation

    new_bboxes = np.array(new_bboxes)  # Ensure new_bboxes is a 2D array
    
    # Now concatenate new_bboxes, new_scores, and new_class_ids into a single array
    boxes = np.concatenate((new_bboxes, new_scores), axis=1)  # Concatenate along columns
    boxes = np.concatenate((boxes, new_class_ids), axis=1)  # Concatenate along columns

    return boxes





def draw_res(img, boxes):
    '''
    在图像上绘制检测结果：
    1. 绘制边界框
    2. 添加类别标签和置信度
    img: 原始图像
    boxes: 检测框信息，包含坐标、置信度和类别ID
    '''
    img = img.astype(np.uint8)  # 确保图像类型正确
    for i, [x, y, w, h, scores, class_ids] in enumerate(boxes):
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        name = coco_class[int(class_ids)]  # 获取类别名称
        print(i + 1, [x, y, w, h], round(scores, 4), name)  # 打印检测信息
        
        label = f'{name} ({scores:.2f})'  # 构建标签文本
        W, H = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]  # 获取文本尺寸
        color = colors[name]  # 获取类别对应的颜色
        
        # 绘制边界框
        cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), color, thickness=2)
        
        # 绘制标签背景
        cv2.rectangle(img, (x, int(y - H)), (int(x + W / 2), y), (0, 255,), -1, cv2.LINE_AA)
        
        # 添加标签文本
        cv2.putText(img, label, (x, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return img
 
# 修改 main 函数
def main(args):
    print("Start video inference...")

    # 初始化模型部分与原代码相同
    size = 640
    config = aidlite.Config.create_instance()
    if config is None:
        print("Create config failed !")
        return False

    config.implement_type = aidlite.ImplementType.TYPE_LOCAL

    if args.model_type.lower() == "qnn":
        config.framework_type = aidlite.FrameworkType.TYPE_QNN231
    elif args.model_type.lower() == "snpe2" or args.model_type.lower() == "snpe":
        config.framework_type = aidlite.FrameworkType.TYPE_SNPE2

    config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
    config.is_quantify_model = 1

    model = aidlite.Model.create_instance(args.target_model)
    if model is None:
        print("Create model failed !")
        return False
    input_shapes = [[1, size, size, 3]]
    output_shapes = [[1, 4, 8400], [1, 80, 8400]]
    
    model.set_model_properties(input_shapes, aidlite.DataType.TYPE_FLOAT32,
                               output_shapes, aidlite.DataType.TYPE_FLOAT32)

    interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
    if interpreter is None:
        print("build_interpretper_from_model_and_config failed !")
        return None
    result = interpreter.init()
    if result != 0:
        print(f"interpreter init failed !")
        return False
    result = interpreter.load_model()
    if result != 0:
        print("interpreter load model failed !")
        return False
    print("Model loaded successfully!")

        # 视频读取
        
    # 初始化三个视频流
    cap1 = cv2.VideoCapture(args.video_path_1)
    cap2 = cv2.VideoCapture(args.video_path_2)
    cap3 = cv2.VideoCapture(args.video_path_3)

    if not cap1.isOpened() or not cap2.isOpened() or not cap3.isOpened():
        print("Error: Could not open one or more video files")
        return False

    # 性能测试
    invoke_times = []
    postprocess_times = []

    print(f"Running video inference...")

    # 用于计算FPS
    frame_counter = 0
    total_invoke_time = 0
    total_postprocess_time = 0

    while cap1.isOpened() and cap2.isOpened() and cap3.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        if not ret1 or not ret2 or not ret3:
            break

        # 记录循环开始时间
        loop_start_time = time.time()

        # 记录每个视频流的检测结果
        detect1, detect2, detect3 = None, None, None
        # 处理每个视频流
        for i, (cap, frame) in enumerate([(cap1, frame1), (cap2, frame2), (cap3, frame3)]):
            # 图像预处理
            img_processed = np.copy(frame)
            [h, w, _] = img_processed.shape
            scale = max((h, w)) / size
            ratio = [scale, scale]

            image = np.zeros((max(h, w), max(h, w), 3), np.uint8)
            image[0:h, 0:w] = img_processed
            img_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_input = cv2.resize(img_input, (size, size))

            mean_data = [0, 0, 0]
            std_data = [255, 255, 255]
            img_input = (img_input - mean_data) / std_data
            img_input = img_input.astype(np.float32)

            # 设置输入tensor
            interpreter.set_input_tensor(0, img_input.data)

            # 只计算模型推理的时间
            t1 = time.time()
            result = interpreter.invoke()
            t2 = time.time()

            if result != 0:
                print("interpreter invoke() failed")
                return False

            invoke_time = (t2 - t1) * 1000
            invoke_times.append(invoke_time)

            # 后处理时间记录
            t1_post = time.time()

            # 检测模型类型并调整输出张量顺序
            if '8550' in args.target_model:
                # 8550模型：tensor 0是类别置信度，tensor 1是边界框
                qnn_conf = interpreter.get_output_tensor(0).reshape(*output_shapes[1])
                qnn_local = interpreter.get_output_tensor(1).reshape(*output_shapes[0])
            else:
                # 6490模型：tensor 0是边界框，tensor 1是类别置信度
                qnn_local = interpreter.get_output_tensor(0).reshape(*output_shapes[0])
                qnn_conf = interpreter.get_output_tensor(1).reshape(*output_shapes[1])
            qnn_result = np.concatenate((qnn_local, qnn_conf), axis=1)
            qnn_result = qnn_result.transpose(0, 2, 1)
            qnn_result = qnn_result[0]

            # 应用后处理，针对每个视频流独立处理
            if i == 0:
                detect1 = postprocess(qnn_result, ratio, conf_threshold=0.25, nms_threshold=0.25)
            elif i == 1:
                detect2 = postprocess(qnn_result, ratio, conf_threshold=0.25, nms_threshold=0.25)
            else:
                detect3 = postprocess(qnn_result, ratio, conf_threshold=0.25, nms_threshold=0.25)

            # 后处理结束时间
            t2_post = time.time()
            postprocess_time = (t2_post - t1_post) * 1000  # 转换为毫秒
            postprocess_times.append(postprocess_time)

            # 绘制结果
            if i == 0:
                res_frame1 = draw_res(frame, list(detect1))
            elif i == 1:
                res_frame2 = draw_res(frame, list(detect2))
            else:
                res_frame3 = draw_res(frame, list(detect3))

        # 记录循环结束时间
        loop_end_time = time.time()

        # 计算该次循环的总时间
        loop_time = (loop_end_time - loop_start_time) * 1000  # 转换为毫秒
        print(f"Loop {frame_counter}: Time taken for processing all three streams: {loop_time:.2f} ms")

        
        # 调整每个视频的大小，适应屏幕显示
        res_frame1 = cv2.resize(res_frame1, (640, 360))  # 调整为较小的尺寸
        res_frame2 = cv2.resize(res_frame2, (640, 360))  # 调整为较小的尺寸
        res_frame3 = cv2.resize(res_frame3, (640, 360))  # 调整为较小的尺寸

        # 合并三个视频成品字结构
        top_row = np.hstack((res_frame1, res_frame2))  # 上排两路视频
        bottom_row = np.hstack((np.zeros_like(res_frame3), res_frame3))  # 下排一路视频，左边空白
        combined = np.vstack((top_row, bottom_row))  # 拼接成品字形

        # 在显示窗口上标注 FPS
        cv2.putText(combined, f"FPS: {1000 / invoke_time:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示合并的三个视频
        cv2.imshow('Three Video Streams', combined)

        # 计算FPS
        frame_counter += 1
        total_invoke_time += invoke_time
        total_postprocess_time += postprocess_time

        if frame_counter > 0:
            avg_invoke_time = total_invoke_time / frame_counter
            avg_postprocess_time = total_postprocess_time / frame_counter
            fps = 1000 / avg_invoke_time  # FPS是1秒内能处理多少帧

            # 每处理一帧打印推理时间、后处理时间和FPS
            print(f"Frame {frame_counter}: "
                  f"Inference Time: {invoke_time:.2f} ms | Postprocess Time: {postprocess_time:.2f} ms | FPS: {fps:.2f}")

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 打印统计信息
    mean_invoke_time = np.mean(invoke_times)
    mean_postprocess_time = np.mean(postprocess_times)
    fps = 1000 / mean_invoke_time

    # 打印性能结果
    print(f"\nTotal {frame_counter} frames processed\n"
          f"Mean inference time: {mean_invoke_time:.2f} ms\n"
          f"Mean postprocess time: {mean_postprocess_time:.2f} ms\n"
          f"FPS: {fps:.2f}\n")

    # 释放资源
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 修改后的ArgumentParser部分
    parser = argparse.ArgumentParser(description="Run video inference benchmarks")
    parser.add_argument('--target_model', type=str,
                        default='yolov8_video-main/model/6490/cutoff_yolov8s_qcs6490_w8a8.qnn231.ctx.bin',
                        help="inference model path")
    
    # 添加三个视频路径参数
    parser.add_argument('--video_path_1', type=str, default='yolov8_video-main/video/1.mp4', help="Input video path 1")
    parser.add_argument('--video_path_2', type=str, default='yolov8_video-main/video/2.mp4', help="Input video path 2")
    parser.add_argument('--video_path_3', type=str, default='yolov8_video-main/video/3.mp4', help="Input video path 3")

    parser.add_argument('--model_type', type=str, default='QNN', help="run backend")
    args = parser.parse_args()
    main(args)
