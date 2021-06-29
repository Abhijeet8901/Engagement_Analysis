import cv2
import numpy as np
from numpy import sin, cos, pi, arctan
from numpy import frombuffer, uint8, concatenate, float32, maximum, minimum, prod
from numpy import zeros, concatenate, float32, tile, repeat, arange, exp
from numpy.linalg import norm
import time
from queue import Queue, Full
from threading import Thread
import sys
import collections
import mxnet as mx
from mxnet.ndarray import waitall, concat
from functools import partial
import os
import tensorflow as tf
from google.colab.patches import cv2_imshow

class HeadPoseEstimator:

    def __init__(self, filepath, W, H) -> None:
        _predefined = np.load(filepath, allow_pickle=True)
        self.object_pts, self.r_vec, self.t_vec = _predefined
        self.cam_matrix = np.array([[W, 0, W/2.0],
                                    [0, W, H/2.0],
                                    [0, 0, 1]])

        self.origin_width = 144.76935
        self.origin_height = 139.839

    def get_head_pose(self, shape):
        if len(shape) == 68:
            image_pts = shape
        elif len(shape) == 106:
            image_pts = shape[[
                9, 10, 11, 14, 16, 3, 7, 8, 0,
                24, 23, 19, 32, 30, 27, 26, 25,
                43, 48, 49, 51, 50, 102, 103, 104, 105, 101,
                72, 73, 74, 86, 78, 79, 80, 85, 84,
                35, 41, 42, 39, 37, 36, 89, 95, 96, 93, 91, 90,
                52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55,
                65, 66, 62, 70, 69, 57, 60, 54
            ]]

            # center = image_pts.mean(axis=0)
            # top_center = shape[[49, 104]].mean(axis=0)

            # left_width = -np.linalg.norm(shape[13]- center)
            # right_width = np.linalg.norm(shape[29]- center)
            # top_height = -np.linalg.norm(top_center- center)
            # bottom_height = np.linalg.norm(shape[0]- center)

            # wfactor = self.origin_width / (right_width - left_width)
            # hfactor = self.origin_height / (bottom_height - top_height)
        else:
            raise RuntimeError('Unsupported shape format')

        # start_time = time.perf_counter()

        ret, rotation_vec, translation_vec = cv2.solvePnP(
            self.object_pts,
            image_pts,
            cameraMatrix=self.cam_matrix,
            distCoeffs=None,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        rear_size = 100
        rear_depth = -200
        front_depth = 0

        # left_width *= wfactor
        # right_width *= wfactor
        # bottom_height *= hfactor
        # top_height *= hfactor

        left_width = -75
        top_height = -90
        right_width = 75
        bottom_height = 90

        reprojectsrc = np.float32([#[-rear_size, -rear_size, rear_depth],
                                   #[-rear_size, rear_size, rear_depth],
                                   #[rear_size, rear_size, rear_depth],
                                   #[rear_size, -rear_size, rear_depth],
                                   # -------------------------------------
                                   [left_width, bottom_height, front_depth],
                                   [right_width, bottom_height, front_depth],
                                   [right_width, top_height, front_depth],
                                   [left_width, top_height, front_depth]])

        reprojectdst, _ = cv2.projectPoints(reprojectsrc,
                                            rotation_vec,
                                            translation_vec,
                                            self.cam_matrix,
                                            distCoeffs=None)

        # end_time = time.perf_counter()
        # print(end_time - start_time)

        reprojectdst = reprojectdst.transpose((1,0,2)).astype(np.int32)

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        euler_angle = cv2.decomposeProjectionMatrix(pose_mat)[-1]

        return reprojectdst, euler_angle

    @staticmethod
    def draw_head_pose_box(src, pts, color=(0, 255, 255), thickness=2, copy=False):
        if copy:
            src = src.copy()

        cv2.polylines(src, pts, True, color, thickness)
        # cv2.polylines(src, pts[-4:][None, ...], True, (255, 255, 0), thickness)

        # cv2.line(src, tuple(pts[1]), tuple(pts[6]), color, line_width)
        # cv2.line(src, tuple(pts[2]), tuple(pts[7]), color, line_width)
        # cv2.line(src, tuple(pts[3]), tuple(pts[8]), color, line_width)

        return src

pred_type = collections.namedtuple('prediction', ['slice', 'close', 'color'])
pred_types = {'face': pred_type(slice(0, 17), False, (173.91, 198.9, 231.795, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), False, (255., 126.99,  14.025, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), False, (255., 126.99,  14.025, 0.4)),
              'nose': pred_type(slice(27, 31), False, (160,  60.945, 112.965, 0.4)),
              'nostril': pred_type(slice(31, 36), False, (160,  60.945, 112.965, 0.4)),
              'eye1': pred_type(slice(36, 42), True, (151.98, 223.125, 137.955, 0.3)),
              'eye2': pred_type(slice(42, 48), True, (151.98, 223.125, 137.955, 0.3)),
              'lips': pred_type(slice(48, 60), True, (151.98, 223.125, 137.955, 0.3)),
              'teeth': pred_type(slice(60, 68), True, (151.98, 223.125, 137.955, 0.4))}


class BaseAlignmentorModel:
    def __init__(self, prefix, epoch, shape, gpu=-1, verbose=False):
        self._device = gpu
        self._ctx = mx.cpu() if self._device < 0 else mx.gpu(self._device)

        self.model = self._load_model(prefix, epoch, shape)
        self.exec_group = self.model._exec_group

        self.input_shape = shape[-2:]
        self.pre_landmarks = None

    def _load_model(self, prefix, epoch, shape):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        model = mx.mod.Module(sym, context=self._ctx, label_names=None)
        model.bind(data_shapes=[('data', shape)], for_training=False)
        model.set_params(arg_params, aux_params)
        return model

    @staticmethod
    def draw_poly(src, landmarks, stroke=1, color=(125, 255, 125), copy=True):
        draw = src.copy() if copy else src

        for pred in pred_types.values():
            le = [landmarks[pred.slice].reshape(-1, 1, 2).astype(np.int32)]
            cv2.polylines(draw, le, pred.close, pred.color, thickness=stroke)

        return draw


class CoordinateAlignmentModel(BaseAlignmentorModel):
    def __init__(self, prefix, epoch, gpu=-1, verbose=False):
        shape = (1, 3, 192, 192)
        super().__init__(prefix, epoch, shape, gpu, verbose)
        self.trans_distance = self.input_shape[-1] >> 1
        self.marker_nums = 106
        self.eye_bound = ([35, 41, 40, 42, 39, 37, 33, 36],
                          [89, 95, 94, 96, 93, 91, 87, 90])

    def _preprocess(self, img, bbox):
        maximum_edge = max(bbox[2:4] - bbox[:2]) * 3.0
        scale = (self.trans_distance << 2) / maximum_edge
        center = (bbox[2:4] + bbox[:2]) / 2.0
        cx, cy = self.trans_distance - scale * center

        M = np.array([[scale, 0, cx], [0, scale, cy]])

        corpped = cv2.warpAffine(img, M, self.input_shape, borderValue=0.0)
        inp = corpped[..., ::-1].transpose(2, 0, 1)[None, ...]

        return mx.nd.array(inp), M

    def _inference(self, x):
        self.exec_group.data_arrays[0][0][1][:] = x.astype(np.float32)
        self.exec_group.execs[0].forward(is_train=False)
        return self.exec_group.execs[0].outputs[-1][-1]

    def _postprocess(self, out, M):
        iM = cv2.invertAffineTransform(M)
        col = np.ones((self.marker_nums, 1))

        out = out.reshape((self.marker_nums, 2))

        pred = out.asnumpy()
        pred += 1
        pred *= self.trans_distance

        # add a column
        # pred = np.c_[pred, np.ones((pred.shape[0], 1))]
        pred = np.concatenate((pred, col), axis=1)
        
        return pred @ iM.T  # dot product

    def _calibrate(self, pred, thd):
        if self.pre_landmarks is not None:
            for i in range(self.marker_nums):
                if sum(abs(self.pre_landmarks[i] - pred[i]) < thd) != 2:
                    self.pre_landmarks[i] = pred[i]
        else:
            self.pre_landmarks = pred

        return self.pre_landmarks

    def get_landmarks(self, image, detected_faces=None, calibrate=False):
        """Predict the landmarks for each face present in the image.
        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.
        Arguments:
            image {numpy.array} -- The input image.
        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None}, format: {x1, y1, x2, y2, score})
        """

        for bbox in detected_faces:
            inp, M = self._preprocess(image, bbox)
            out = self._inference(inp)
            pred = self._postprocess(out, M)

            yield self._calibrate(pred, .8) if calibrate else pred

class AnchorConifg:
    def __init__(self, *,  stride, scales,
                 base_size=16, ratios=(1., ), dense_anchor=False):
        self.stride = stride
        self.scales = np.array(scales)
        self.scales_shape = self.scales.shape[0]

        self.base_size = base_size
        self.ratios = np.array(ratios)
        self.dense_anchor = dense_anchor

        self.base_anchors = self._generate_anchors()

    def _generate_anchors(self):
        base_anchor = np.array([1, 1, self.base_size, self.base_size]) - 1
        ratio_anchors = self._ratio_enum(base_anchor)

        anchors = np.vstack([self._scale_enum(ratio_anchors[i, :])
                             for i in range(ratio_anchors.shape[0])])

        if self.dense_anchor:
            assert self.stride % 2 == 0
            anchors2 = anchors.copy()
            anchors2[:, :] += int(self.stride/2)
            anchors = np.vstack((anchors, anchors2))

        return anchors

    def _whctrs(self, anchor):
        """
        Return width, height, x center, and y center for an anchor (window).
        """

        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    def _mkanchors(self, ws, hs, x_ctr, y_ctr):
        """
        Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """

        ws = ws[:, None]
        hs = hs[:, None]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)))
        return anchors

    def _ratio_enum(self, anchor):
        """
        Enumerate a set of anchors for each aspect ratio wrt an anchor.
        """

        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        size = w * h
        size_ratios = size / self.ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * self.ratios)
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def _scale_enum(self, anchor):
        """
        Enumerate a set of anchors for each scale wrt an anchor.
        """

        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        ws = w * self.scales
        hs = h * self.scales
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def __repr__(self):
        return f'Stride: {self.stride}'


anchor_config = [
    AnchorConifg(stride=32, scales=(32, 16)),
    AnchorConifg(stride=16, scales=(8, 4)),
    # AnchorConifg(stride=8, scales=(2, 1)),
]


def generate_runtime_anchors(height, width, stride, base_anchors):
    A = base_anchors.shape[0]

    all_anchors = zeros((height*width, A, 4), dtype=float32)

    rw = tile(arange(0, width*stride, stride),
              height).reshape(-1, 1, 1)
    rh = repeat(arange(0, height*stride, stride),
                width).reshape(-1, 1, 1)

    all_anchors += concatenate((rw, rh, rw, rh), axis=2)
    all_anchors += base_anchors

    return all_anchors


def generate_anchors_fpn(dense_anchor=False, cfg=anchor_config):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    return sorted(cfg, key=lambda x: x.stride, reverse=True)


def nonlinear_pred(boxes, box_deltas):
    if boxes.size:
        ctr_x, ctr_y, widths, heights = boxes.T
        widths -= ctr_x
        heights -= ctr_y

        widths += 1.0
        heights += 1.0

        dx, dy, dw, dh, _ = box_deltas.T

        dx *= widths
        dx += ctr_x
        dx += 0.5 * widths

        dy *= heights
        dy += ctr_y
        dy += 0.5 * heights

        exp(dh, out=dh)
        dh *= heights
        dh -= 1.0
        dh *= 0.5

        exp(dw, out=dw)
        dw *= widths
        dw -= 1.0
        dw *= 0.5

        dx -= dw
        dw += dw
        dw += dx

        dy -= dh
        dh += dh
        dh += dy

class BaseDetection:
    def __init__(self, *, thd, gpu, margin, nms_thd, verbose):
        self.threshold = thd
        self.nms_threshold = nms_thd
        self.device = gpu
        self.margin = margin

        self._queue = Queue(200)
        self.write_queue = self._queue.put_nowait
        self.read_queue = iter(self._queue.get, b'')

        self._nms_wrapper = partial(self.non_maximum_suppression,
                                    threshold=self.nms_threshold)
        
        self._biggest_wrapper = partial(self.find_biggest_box)


    def margin_clip(self, b):
        margin_x = (b[2] - b[0]) * self.margin
        margin_y = (b[3] - b[1]) * self.margin

        b[0] -= margin_x
        b[1] -= margin_y
        b[2] += margin_x
        b[3] += margin_y

        return np.clip(b, 0, None, out=b)

    @staticmethod
    def find_biggest_box(dets):
        return max(dets, key=lambda x: x[4]) if dets.size > 0 else None

    @staticmethod
    def non_maximum_suppression(dets, threshold):
        ''' ##### Author 1996scarlet@gmail.com
        Greedily select boxes with high confidence and overlap with threshold.
        If the boxes' overlap > threshold, we consider they are the same one.
        Parameters
        ----------
        dets: ndarray
            Bounding boxes of shape [N, 5].
            Each box has [x1, y1, x2, y2, score].
        threshold: float
            The src scales para.
        Returns
        -------
        Generator of kept box, each box has [x1, y1, x2, y2, score].
        Usage
        -----
        >>> for res in non_maximum_suppression(dets, thresh):
        >>>     pass
        '''

        x1, y1, x2, y2, scores = dets.T

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        while order.size > 0:
            keep, others = order[0], order[1:]

            yield np.copy(dets[keep])

            xx1 = maximum(x1[keep], x1[others])
            yy1 = maximum(y1[keep], y1[others])
            xx2 = minimum(x2[keep], x2[others])
            yy2 = minimum(y2[keep], y2[others])

            w = maximum(0.0, xx2 - xx1 + 1)
            h = maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            overlap = inter / (areas[keep] - inter + areas[others])

            order = others[overlap < threshold]

    @staticmethod
    def filter_boxes(boxes, min_size, max_size=-1):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        if max_size > 0:
            boxes = np.where(minimum(ws, hs) < max_size)[0]
        if min_size > 0:
            boxes = np.where(maximum(ws, hs) > min_size)[0]
        return boxes


class MxnetDetectionModel(BaseDetection):
    def __init__(self, prefix, epoch, scale=1., gpu=-1, thd=0.6, margin=0,
                 nms_thd=0.4, verbose=False):

        super().__init__(thd=thd, gpu=gpu, margin=margin,
                         nms_thd=nms_thd, verbose=verbose)

        self.scale = scale
        self._rescale = partial(cv2.resize, dsize=None, fx=self.scale,
                                fy=self.scale, interpolation=cv2.INTER_NEAREST)

        self._ctx = mx.cpu() if self.device < 0 else mx.gpu(self.device)
        self._fpn_anchors = generate_anchors_fpn()
        self._runtime_anchors = {}

        self.model = self._load_model(prefix, epoch)
        self.exec_group = self.model._exec_group

    def _load_model(self, prefix, epoch):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        model = mx.mod.Module(sym, context=self._ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, 1, 1))],
                   for_training=False)
        model.set_params(arg_params, aux_params)
        return model

    def _get_runtime_anchors(self, height, width, stride, base_anchors):
        key = height, width, stride
        if key not in self._runtime_anchors:
            self._runtime_anchors[key] = generate_runtime_anchors(
                height, width, stride, base_anchors).reshape((-1, 4))
        return self._runtime_anchors[key]

    def _retina_detach(self, out):
        ''' ##### Author 1996scarlet@gmail.com
        Solving bounding boxes.
        Parameters
        ----------
        out: map object of staggered scores and deltas.
            scores, deltas = next(out), next(out)
            Each scores has shape [N, A*4, H, W].
            Each deltas has shape [N, A*4, H, W].
            N is the batch size.
            A is the shape[0] of base anchors declared in the fpn dict.
            H, W is the heights and widths of the anchors grid,
            based on the stride and input image's height and width.
        Returns
        -------
        Generator of list, each list has [boxes, scores].
        Usage
        -----
        >>> np.block(list(self._retina_solving(out)))
        '''

        buffer, anchors = out[0].asnumpy(), out[1]
        mask = buffer[:, 4] > self.threshold
        deltas = buffer[mask]
        nonlinear_pred(anchors[mask], deltas)
        deltas[:, :4] /= self.scale
        return deltas

    def _retina_solve(self):
        out, res, anchors = iter(self.exec_group.execs[0].outputs), [], []

        for fpn in self._fpn_anchors:
            scores = next(out)[:, -fpn.scales_shape:,
                               :, :].transpose((0, 2, 3, 1))
            deltas = next(out).transpose((0, 2, 3, 1))

            res.append(concat(deltas.reshape((-1, 4)),
                              scores.reshape((-1, 1)), dim=1))

            anchors.append(self._get_runtime_anchors(*deltas.shape[1:3],
                                                     fpn.stride,
                                                     fpn.base_anchors))

        return concat(*res, dim=0), concatenate(anchors)

    def _retina_forward(self, src):
        ''' ##### Author 1996scarlet@gmail.com
        Image preprocess and return the forward results.
        Parameters
        ----------
        src: ndarray
            The image batch of shape [H, W, C].
        scales: list of float
            The src scales para.
        Returns
        -------
        net_out: list, len = STEP * N
            If step is 2, each block has [scores, bbox_deltas]
            Else if step is 3, each block has [scores, bbox_deltas, landmarks]
        Usage
        -----
        >>> out = self._retina_forward(frame)
        '''
        # timea = time.perf_counter()

        dst = self._rescale(src).transpose((2, 0, 1))[None, ...]

        if dst.shape != self.model._data_shapes[0].shape:
            self.exec_group.reshape([mx.io.DataDesc('data', dst.shape)], None)

        self.exec_group.data_arrays[0][0][1][:] = dst.astype(float32)
        self.exec_group.execs[0].forward(is_train=False)

        # print(f'inferance: {time.perf_counter() - timea}')

        return self._retina_solve()

    def detect(self, image, mode='nms'):
        out = self._retina_forward(image)
        detach = self._retina_detach(out)
        return getattr(self, f'_{mode}_wrapper')(detach)

    def workflow_inference(self, instream, shape):
        for source in instream:
            # st = time.perf_counter()

            frame = frombuffer(source, dtype=uint8).reshape(shape)

            out = self._retina_forward(frame)

            try:
                self.write_queue((frame, out))
            except Full:
                waitall()
                print('Frame queue full', file=sys.stderr)

            # print(f'workflow_inference: {time.perf_counter() - st}')

    def workflow_postprocess(self, outstream=None):
        for frame, out in self.read_queue:
            # st = time.perf_counter()
            detach = self._retina_detach(out)
            # print(f'workflow_postprocess: {time.perf_counter() - st}')

            if outstream is None:
                for res in self._nms_wrapper(detach):
                    # self.margin_clip(res)
                    cv2.rectangle(frame, (res[0], res[1]),
                                  (res[2], res[3]), (255, 255, 0))

                cv2.imshow('res', frame)
                cv2.waitKey(1)
            else:
                outstream(frame)
                outstream(detach)

class IrisLocalizationModel():

    def __init__(self, filepath):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=filepath)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.trans_distance = 32
        self.input_shape = (64, 64)

    def _preprocess(self, img, length, center, name=None):
        """Preprocess the image to meet the model's input requirement.
        Args:
            img: An image in default BGR format.
        Returns:
            image_norm: The normalized image ready to be feeded.
        """

        scale = 23 / length
        cx, cy = self.trans_distance - scale * center

        M = np.array([[scale, 0, cx], [0, scale, cy]])

        resized = cv2.warpAffine(img, M, self.input_shape, borderValue=0.0)

        if name is not None:
            cv2.imshow(name, resized)

        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32)
        cv2.normalize(image_norm, image_norm, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX) 

        return image_norm, M

    def get_mesh(self, image, length, center, name=None):
        """Detect the face mesh from the image given.
        Args:
            image: An image in default BGR format.
        Returns:
            mesh: An eyebrow mesh, normalized.
            iris: Iris landmarks.
        """

        # Preprocess the image before sending to the network.
        image, M = self._preprocess(image, length, center, name)

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image[tf.newaxis, :]

        # The actual detection.
        self.interpreter.set_tensor(self.input_details[0]["index"], image)
        self.interpreter.invoke()

        # Save the results.
        # mesh = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        iris = self.interpreter.get_tensor(self.output_details[1]["index"])[0]

        iris = iris.reshape(-1, 3)
        iris[:, 2] = 1
        iM = cv2.invertAffineTransform(M)

        return iris @ iM.T

    @staticmethod
    def draw_pupil(iris, frame, color=(0, 0, 255), thickness=2):
        pupil = iris[0]
        radius = np.linalg.norm(iris[1:] - iris[0], axis=1)

        pupil = pupil.astype(int)
        radius = int(max(radius))

        cv2.circle(frame, tuple(pupil), radius, color, thickness, cv2.LINE_AA)

        return pupil, radius

    @staticmethod
    def draw_eye_markers(landmarks, frame, close=True, color=(0, 255, 255), thickness=2):
        landmarks = landmarks.astype(np.int32)
        cv2.polylines(frame, landmarks, close, color, thickness, cv2.LINE_AA)

SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)

def calculate_3d_gaze(frame, poi, scale=256):
    starts, ends, pupils, centers = poi

    eye_length = norm(starts - ends, axis=1)
    ic_distance = norm(pupils - centers, axis=1)
    zc_distance = norm(pupils - starts, axis=1)

    s0 = (starts[:, 1] - ends[:, 1]) * pupils[:, 0]
    s1 = (starts[:, 0] - ends[:, 0]) * pupils[:, 1]
    s2 = starts[:, 0] * ends[:, 1]
    s3 = starts[:, 1] * ends[:, 0]

    delta_y = (s0 - s1 + s2 - s3) / eye_length / 2
    delta_x = np.sqrt(abs(ic_distance**2 - delta_y**2))

    delta = np.array((delta_x * SIN_LEFT_THETA,
                      delta_y * SIN_UP_THETA))
    delta /= eye_length
    theta, pha = np.arcsin(delta)

    # print(f"THETA:{180 * theta / pi}, PHA:{180 * pha / pi}")
    # delta[0, abs(theta) < 0.1] = 0
    # delta[1, abs(pha) < 0.03] = 0

    inv_judge = zc_distance**2 - delta_y**2 < eye_length**2 / 4

    delta[0, inv_judge] *= -1
    theta[inv_judge] *= -1
    delta *= scale

    # cv2.circle(frame, tuple(pupil.astype(int)), 2, (0, 255, 255), -1)
    # cv2.circle(frame, tuple(center.astype(int)), 1, (0, 0, 255), -1)

    return theta, pha, delta.T


def draw_sticker(src, offset, pupils, landmarks,status,blink_thd=0.22,arrow_color=(0, 125, 255), copy=False):
    if copy:
        src = src.copy()

    left_eye_hight = landmarks[33, 1] - landmarks[40, 1]
    left_eye_width = landmarks[39, 0] - landmarks[35, 0]

    right_eye_hight = landmarks[87, 1] - landmarks[94, 1]
    right_eye_width = landmarks[93, 0] - landmarks[89, 0]

    for mark in landmarks.reshape(-1, 2).astype(int):
        cv2.circle(src, tuple(mark), radius=1,
                   color=(0, 0, 255), thickness=-1)

    left_eye_closed,right_eye_closed,both_closed=False,False,False 

    if left_eye_hight / left_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(pupils[0].astype(int)),
                        tuple((offset+pupils[0]).astype(int)), arrow_color, 2)
    else:
        left_eye_closed=True

    if right_eye_hight / right_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(pupils[1].astype(int)),
                        tuple((offset+pupils[1]).astype(int)), arrow_color, 2)
    else:
        right_eye_closed=True
    
    if left_eye_closed and right_eye_closed:         ###### Checking if both eyes closed ##############
      both_closed=True

    return src,both_closed


def main(video,max_eye_offset=25,gpu_ctx=-1):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    calc_timestamps = [0.0]
    engagement_status=[]

    fd = MxnetDetectionModel("weights/16and32", 0, .6, gpu=gpu_ctx)
    fa = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
    gs = IrisLocalizationModel("weights/iris_landmark.tflite")
    hp = HeadPoseEstimator("weights/object_points.npy", cap.get(3), cap.get(4))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    use_past_k_frames=int(fps/2)          ###### no of past frames to take average of for engagement #######
    if use_past_k_frames%2==0:
        use_past_k_frames+=1
    past_k_statuses=[]       ###### storing those past statuses ##########
    #output_video,output_path=True,"Result.mp4"
    #if output_video==True:
    #    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #    writer = cv2.VideoWriter(output_path,fourcc, fps, (width,height),True)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        bboxes = fd.detect(frame)
        landmark_detected=False
        for landmarks in fa.get_landmarks(frame, bboxes, calibrate=True):
            # calculate head pose
            landmark_detected=True
            _, euler_angle = hp.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            eye_markers = np.take(landmarks, fa.eye_bound, axis=0)
            
            eye_centers = np.average(eye_markers, axis=1)
            eye_lengths = (landmarks[[39, 93]] - landmarks[[35, 89]])[:, 0]
            iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
            pupil_left, _ = gs.draw_pupil(iris_left, frame, thickness=1)

            iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
            pupil_right, _ = gs.draw_pupil(iris_right, frame, thickness=1)

            pupils = np.array([pupil_left, pupil_right])

            poi = landmarks[[35, 89]], landmarks[[39, 93]], pupils, eye_centers
            theta, pha, delta = calculate_3d_gaze(frame, poi)

            if yaw > 30:
                end_mean = delta[0]
            elif yaw < -30:
                end_mean = delta[1]
            else:
                end_mean = np.average(delta, axis=0)

            if end_mean[0] < 0:
                zeta = arctan(end_mean[1] / end_mean[0]) + pi
            else:
                zeta = arctan(end_mean[1] / (end_mean[0] + 1e-7))

            if roll < 0:
                roll += 180
            else:
                roll -= 180

            real_angle = zeta + roll * pi / 180

            R = norm(end_mean)
            offset = R * cos(real_angle), R * sin(real_angle)
            
            eye_offset = round(R*cos(real_angle),4)         ########### Calculating Eye Offset for predicting Engagement ######################
            status=""
            #if output_video==True:
            #    cv2.putText(frame,str(eye_offset),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
            
            if abs(eye_offset)<max_eye_offset:                              ########### If Absolute Value of Eye Offset is less than 25 then we its engaged,else not ###########
              status="Engaged"
            else:
              status="DisEngaged"

            landmarks[[38, 92]] = landmarks[[34, 88]] = eye_centers

            _,both_closed=draw_sticker(frame, offset, pupils, landmarks,status)
            if both_closed:                              ########## Also if both eyes are closed, we say the user is DisEngaged ############################
              status="DisEngaged"

        if len(past_k_statuses)<use_past_k_frames:       #### Updating past_k_statuses for taking average of last k frames #####
          past_k_statuses.append(status)
        else:
          del past_k_statuses[0]
          past_k_statuses.append(status)
            
        no_eng,no_dis=0,0
        for i in past_k_statuses:
          if i=="Engaged":
            no_eng+=1
          else:
            no_dis+=1
        
        if landmark_detected==False:
          engagement_status.append("DisEngaged")
        else:
          if no_eng>no_dis:
            engagement_status.append("Engaged")
            #if output_video==True:
            #    cv2.putText(frame,"Engaged",(300,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
          elif no_eng<no_dis:
            engagement_status.append("DisEngaged")
            #if output_video==True:
            #    cv2.putText(frame,"DisEngaged",(300,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
          else:
            engagement_status.append(status)
            #if output_video==True:
            #    if status=="Engaged":
            #        cv2.putText(frame,"Engaged",(300,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
            #    else:
            #        cv2.putText(frame,"DisEngaged",(300,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
        calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
        
        #if output_video==True:
        #    writer.write(frame)
    
    #if output_video==True:
    #    writer.release()
    cap.release()
    final_output=[]
    for i in range(len(engagement_status)):
      final_output.append([engagement_status[i],calc_timestamps[i],calc_timestamps[i+1]])

    return final_output