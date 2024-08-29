import os
import cv2
import numpy as np
import time
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.utils import platform

if platform == 'android':
    from jnius import autoclass

    File = autoclass('java.io.File')
    Interpreter = autoclass('org.tensorflow.lite.Interpreter')
    InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
    Tensor = autoclass('org.tensorflow.lite.Tensor')
    DataType = autoclass('org.tensorflow.lite.DataType')
    TensorBuffer = autoclass('org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
    ByteBuffer = autoclass('java.nio.ByteBuffer')

    class TensorFlowModel():
        def load(self, model_filename, num_threads=None):
            model = File(model_filename)
            options = InterpreterOptions()
            if num_threads is not None:
                options.setNumThreads(num_threads)
            self.interpreter = Interpreter(model, options)
            self.allocate_tensors()

        def allocate_tensors(self):
            self.interpreter.allocateTensors()
            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.output_shape = self.interpreter.getOutputTensor(0).shape()
            self.output_type = self.interpreter.getOutputTensor(0).dataType()

        def get_input_shape(self):
            return self.input_shape

        def resize_input(self, shape):
            if self.input_shape != shape:
                self.interpreter.resizeInput(0, shape)
                self.allocate_tensors()

        def pred(self, x):
            input = ByteBuffer.wrap(x.tobytes())
            output = TensorBuffer.createFixedSize(self.output_shape, self.output_type)
            self.interpreter.run(input, output.getBuffer().rewind())
            return np.reshape(np.array(output.getFloatArray()), self.output_shape)

else:
    import tensorflow as tf

    class TensorFlowModel():
        def load(self, model_filename, num_threads=None):
            self.interpreter = tf.lite.Interpreter(model_filename, num_threads=num_threads)
            self.interpreter.allocate_tensors()

        def resize_input(self, shape):
            if list(self.get_input_shape()) != shape:
                self.interpreter.resize_tensor_input(0, shape)
                self.interpreter.allocate_tensors()

        def get_input_shape(self):
            return self.interpreter.get_input_details()[0]['shape']

        def pred(self, x):
            # assumes one input and one output for now
            self.interpreter.set_tensor(
                self.interpreter.get_input_details()[0]['index'], x)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(
                self.interpreter.get_output_details()[0]['index'])

# Загрузка модели TFLite
model = TensorFlowModel()
model.load(os.path.join(os.getcwd(), '3.tflite'))

counter = 0
stage = "none"
start_time = time.time()

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

def calculate_angle(a, b, c):
    """Calculate the angle between three points a, b, c.
    a, b, c are numpy arrays of shape (2,).
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

class PoseEstimationApp(App):
    def build(self):
        self.img = Image()
        self.reps_label = Label(font_size='20sp', halign='center', valign='middle')
        self.reps_label.bind(size=self.reps_label.setter('text_size'))  # ensure the text wraps inside the label
        self.stage_label = Label(font_size='20sp', halign='center', valign='middle')
        self.stage_label.bind(size=self.stage_label.setter('text_size'))  # ensure the text wraps inside the label
        
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img)
        
        labels_layout = BoxLayout(size_hint_y=None, height=100)
        labels_layout.add_widget(self.reps_label)
        labels_layout.add_widget(self.stage_label)
        
        layout.add_widget(labels_layout)
        
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # обновление 30 раз в секунду
        
        return layout

    def update(self, dt):
        global counter, stage
        ret, frame = self.capture.read()
        if not ret:
            return
        
        img = frame.copy()
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        keypoints_with_scores = model.pred(img)
        
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores, 0.4)
        
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
        left_hip = shaped[11][:2]
        left_knee = shaped[13][:2]
        left_ankle = shaped[15][:2]
        right_hip = shaped[12][:2]
        right_knee = shaped[14][:2]
        right_ankle = shaped[16][:2]
        
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        if time.time() - start_time > 20:
            self.reps_label.text = 'REPS: 0'
            self.stage_label.text = 'STAGE: NONE'
            if left_knee_angle > 160 and right_knee_angle > 160:
                stage = "up"
            if left_knee_angle < 120 and right_knee_angle < 120 and stage == 'up':
                stage = "down"
                counter += 1
            self.reps_label.text = f'REPS: {counter}'
            self.stage_label.text = f'STAGE: {stage}'
        else:
            self.reps_label.text = 'You must be completely visible to the camera, you have 20 seconds'
        frame = cv2.flip(frame, 0)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture

if __name__ == '__main__':
    PoseEstimationApp().run()
