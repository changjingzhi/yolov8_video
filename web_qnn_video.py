import cv2
import numpy as np
import time
import threading
from flask import Flask, Response
import aidlite   # 你已有的库
import argparse
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
        # print(i + 1, [x, y, w, h], round(scores, 4), name)  # 打印检测信息
        
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
 
# ---------------------------
# Flask Web 部分
# ---------------------------
app = Flask(__name__)
outputFrame = None
lock = threading.Lock()

@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Inference Stream</title>
        <style>
            body { text-align: center; background-color: #000; }
            h1 { color: #00ff88; font-family: monospace; }
            img { width: 95%; border-radius: 10px; box-shadow: 0 0 20px #00ff88; }
        </style>
    </head>
    <body>
        <h1>YOLO Real-Time Inference (via Flask)</h1>
        <img src="/video_feed">
    </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    def generate():
        global outputFrame, lock
        while True:
            with lock:
                if outputFrame is None:
                    continue
                flag, encodedImage = cv2.imencode(".jpg", outputFrame)
                if not flag:
                    continue
                frame = encodedImage.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------
# 推理主函数
# ---------------------------
def main(args):
    print("Start video inference...")

    # 启动 Flask 线程（后台运行）
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=14514,
                                            debug=False, use_reloader=False)).start()

    size = 640
    config = aidlite.Config.create_instance()
    if config is None:
        print("Create config failed !")
        return False

    config.implement_type = aidlite.ImplementType.TYPE_LOCAL
    if args.model_type.lower() == "qnn":
        config.framework_type = aidlite.FrameworkType.TYPE_QNN231
    elif args.model_type.lower() in ["snpe2", "snpe"]:
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
    if interpreter.init() != 0 or interpreter.load_model() != 0:
        print("Interpreter init or load model failed !")
        return False
    print("Model loaded successfully!")

    # 打开三个视频
    cap1 = cv2.VideoCapture(args.video_path_1)
    cap2 = cv2.VideoCapture(args.video_path_2)
    cap3 = cv2.VideoCapture(args.video_path_3)
    if not (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
        print("Error: Could not open one or more video files")
        return False

    invoke_times, postprocess_times = [], []
    frame_counter = 0
    print("Running video inference...")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        # 检查视频读取状态
        if not (ret1 and ret2 and ret3):
            print("⚠️ One of the video streams ended or failed to open.")
            break


        loop_start = time.time()

        # 分别推理三路视频
        results = []
        for frame in [frame1, frame2, frame3]:
            img_processed = np.copy(frame)
            h, w, _ = img_processed.shape
            scale = max(h, w) / size
            ratio = [scale, scale]

            image = np.zeros((max(h, w), max(h, w), 3), np.uint8)
            image[:h, :w] = img_processed
            img_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_input = cv2.resize(img_input, (size, size))
            img_input = (img_input / 255.0).astype(np.float32)

            interpreter.set_input_tensor(0, img_input.data)

            t1 = time.time()
            if interpreter.invoke() != 0:
                print("interpreter invoke() failed")
                return False
            t2 = time.time()
            invoke_time = (t2 - t1) * 1000
            invoke_times.append(invoke_time)

            qnn_local = interpreter.get_output_tensor(1).reshape(*output_shapes[0])
            qnn_conf = interpreter.get_output_tensor(0).reshape(*output_shapes[1])
            qnn_result = np.concatenate((qnn_local, qnn_conf), axis=1)
            qnn_result = qnn_result.transpose(0, 2, 1)[0]

            t1_post = time.time()
            detect = postprocess(qnn_result, ratio, conf_threshold=0.25, nms_threshold=0.25)
            t2_post = time.time()
            post_time = (t2_post - t1_post) * 1000
            postprocess_times.append(post_time)

            res_frame = draw_res(frame, list(detect))
            results.append(res_frame)

        # 拼接显示画面
        res_frame1 = cv2.resize(results[0], (640, 360))
        res_frame2 = cv2.resize(results[1], (640, 360))
        res_frame3 = cv2.resize(results[2], (640, 360))
        top_row = np.hstack((res_frame1, res_frame2))
        bottom_row = np.hstack((np.zeros_like(res_frame3), res_frame3))
        combined = np.vstack((top_row, bottom_row))

        fps = 1000.0 / (np.mean(invoke_times[-10:]) + 1e-6)
        cv2.putText(combined, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 更新网页帧
        global outputFrame
        with lock:
            outputFrame = combined.copy()


        frame_counter += 1
        loop_time = (time.time() - loop_start) * 1000
        # print(f"Frame {frame_counter}: Loop {loop_time:.2f} ms | FPS {fps:.2f}")

    # 性能统计
    mean_invoke = np.mean(invoke_times)
    mean_post = np.mean(postprocess_times)
# print(f"\nTotal {frame_counter} frames\n"
#   f"Mean inference: {mean_invoke:.2f} ms | Postprocess: {mean_post:.2f} ms | FPS: {1000/mean_invoke:.2f}")

    cap1.release(), cap2.release(), cap3.release()


# ---------------------------
# 命令行启动参数
# ---------------------------
if __name__ == "__main__":
    # 修改后的ArgumentParser部分
    parser = argparse.ArgumentParser(description="Run video inference benchmarks")
    parser.add_argument('--target_model', type=str,
                        default='model/cutoff_yolov8n_qcs8550_w8a8.qnn231.ctx.bin',
                        help="inference model path")
    
    # 添加三个视频路径参数
    parser.add_argument('--video_path_1', type=str, default='video/1.mp4', help="Input video path 1")
    parser.add_argument('--video_path_2', type=str, default='video/2.mp4', help="Input video path 2")
    parser.add_argument('--video_path_3', type=str, default='video/3.mp4', help="Input video path 3")
    
    parser.add_argument('--model_type', type=str, default='QNN', help="run backend")
    args = parser.parse_args()
    main(args)
