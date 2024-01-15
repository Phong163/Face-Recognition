
import sys
from pathlib import Path
import torch
import sys
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QMovie,QImage, QPixmap
from PyQt5 import QtWidgets
from main import Ui_MainWindow
import argparse
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from config import get_gif
    #detect.py

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt
t=0
class MYUI(Ui_MainWindow):  
    def __init__(self):
        super().__init__()
        self.setupUi(MainWindow)
        self.start.clicked.connect(self.start_conversation)
        self.pushButton.clicked.connect(self.EndWindown)
        self.Page1.clicked.connect(lambda: self.changePage(1))
        self.Page2.clicked.connect(lambda: self.changePage(2))
        self.Page3.clicked.connect(lambda: self.changePage(3))
        self.start_animation()
        self.thread1 = None
        self.thread2 = None
    def start_animation(self):
        gif_path = get_gif("1_1.gif")
        self.movie = QMovie(gif_path)
        self.label.setMovie(self.movie)
        self.movie.setScaledSize(self.label.size()) 
        self.movie.start()
    def start_conversation(self):
        self.start.setStyleSheet("background-color: rgb(0, 170, 255);")
        global t
        t=t+1
        if t==1:
            if not self.thread1 or not self.thread1.isRunning():
                self.thread1 = Task1Thread()
                self.thread1.update_signal.connect(self.change_gif)
                self.thread1.start()
        else:
            print('thread1 dang chay')
    def change_gif(self, gif_path):
        self.movie.stop()
        self.movie.setFileName(gif_path)
        self.movie.start()
    def chan_gif_page2(self,gif_path,index):
        movie = QMovie(get_gif("Spinner.gif"))
        if index==5:
            self.label_5.setMovie(movie)
            self.label_5.setScaledContents(True)
        if index==6:
            self.label_6.setMovie(movie)
            self.label_6.setScaledContents(True)
        movie.start()
    def update_frame(self):
        global im0
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        height, width, channel = im0.shape
        bytesPerLine = 3 * width
        image = QImage(im0.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.label_2.setPixmap(pixmap)
    def changePage(self, index):
        if index==1:
            self.stackedWidget.setCurrentWidget(self.page_1)
        elif index==2:
            self.stackedWidget.setCurrentWidget(self.page_2)
        elif index==3:
            self.stackedWidget.setCurrentWidget(self.page_3)
    def EndWindown(self):
        app.quit()
opt = parse_opt()

class Task1Thread(QThread):
    update_signal = pyqtSignal(str)
    def run(self):
        self.update_signal.emit(get_gif('4.gif'))
        self.detect_face(**vars(opt))
    def detect_face(self,
        weights=ROOT / 'best.pt',  # model path or triton URL
        source=ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        view_img=False,  # show results
        nosave=False,  # do not save images/videos
        save_crop=False,  # save cropped prediction boxes
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        #đọc khung hình từ webcam
        if webcam:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        global im0
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # Inference
            with dt[1]:
                pred = model(im, augment=augment, visualize=visualize)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                x=im.shape[3]
                y=im.shape[2]
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Thay đổi tỷ lệ các hộp từ kích thước img_size thành kích thước im0
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    boxes = det[:, :4].int()
                    labels = det[:, -1].int()
                    for box, label in zip(boxes, labels):
                        x1, y1, x2, y2 = box.tolist()
                        print(f'Label: {label}')
                        if label==0:
                            if 0 <= x1 <0.35*x:
                                print(f' x1: {x1}, y1: {y1}')
                                self.update_signal.emit(get_gif('5.gif'))
                                print('đã thêm')
                            elif 0 <= y1 <0.35*y:
                                print(f' x1: {x1}, y1: {y1}')
                                self.update_signal.emit(get_gif('6.gif'))
                                print('đã thêm')
                            elif 0.65*y<= y1 <y:
                                print(f' x1: {x1}, y1: {y1}')
                                self.update_signal.emit(get_gif('7.gif'))
                                print('đã thêm')
                            elif 0.65*x<= x1 <x:
                                print(f' x1: {x1}, y1: {y1}')
                                self.update_signal.emit(get_gif('8.gif'))
                                print('đã thêm')
                            else:
                                print(f' x1: {x1}, y1: {y1}')
                                self.update_signal.emit(get_gif('4.gif'))
                                print('đã thêm')
                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                
                im0 = annotator.result()
                
                ui.update_frame()
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MYUI()
    MainWindow.show()
    sys.exit(app.exec_())

