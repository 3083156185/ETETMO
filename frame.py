import cv2
import numpy as np

def frame_difference(prev_frame, cur_frame, next_frame):
    """
    计算三帧差分法的差分结果。
    """
    # 计算前一帧与当前帧的差分
    diff_frames1 = cv2.absdiff(prev_frame, cur_frame)

    # 计算当前帧与后一帧的差分
    diff_frames2 = cv2.absdiff(cur_frame, next_frame)

    # 结合两次差分结果
    result = cv2.bitwise_and(diff_frames1, diff_frames2)

    return result

def main(show_video=False):
    # 打开视频文件或摄像头
    cap = cv2.VideoCapture('1.mp4')  # 或者用 0 表示摄像头

    # 获取视频的宽度、高度和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 定义视频编解码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器保存为 .mp4 文件
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height), isColor=True)

    # 读取前三帧
    ret, prev_frame = cap.read()
    ret, cur_frame = cap.read()
    ret, next_frame = cap.read()

    if not ret:
        print("无法读取视频")
        return

    # 转换为灰度图
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # 计算三帧差分
        diff = frame_difference(prev_frame_gray, cur_frame_gray, next_frame_gray)

        # 应用阈值以使运动目标更明显
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        # 将阈值图像转换为伪彩色图像
        color_mapped_thresh = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)

        # 创建一个空的黑色图像，用于存放运动目标
        motion_display = np.zeros_like(cur_frame)

        # 将伪彩色的运动区域复制到 motion_display 中
        motion_display[thresh == 255] = color_mapped_thresh[thresh == 255]

        # 将结果写入视频文件
        out.write(motion_display)

        # 显示差分结果（如果需要）
        if show_video:
            cv2.imshow("Motion Detection", motion_display)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        # 更新帧
        prev_frame_gray = cur_frame_gray
        cur_frame_gray = next_frame_gray
        prev_frame = cur_frame
        cur_frame = next_frame

        # 读取下一帧
        ret, next_frame = cap.read()

        if not ret:
            break

        # 转换为灰度图
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 释放摄像头或视频文件
    cap.release()
    out.release()  # 释放 VideoWriter 对象
    if show_video:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(show_video=False)  # 将 show_video 设置为 False 以跳过显示窗口