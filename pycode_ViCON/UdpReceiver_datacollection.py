# Authors: Songnan BAI, Song LI
# Date: 2026-03
# Version: 1.1
# Contact: p.chirarattananon@utoronto.ca
# 
# Description: This script is for the real-time data collection of the hopping robot
# with mocap feedback. Data is saved in a .mat file for later analysis.

import socket
import time
import numpy as np
import threading
from tqdm import tqdm
import struct
import math

from collections import deque
import scipy.io as sio  # save data as a .mat file


class DataSaver(object):
    def __init__(self, *args):
        self.varNameList = args
        self.varNum = len(args)
        self.info_list = {'other_information': -1}
        for i in range(self.varNum):
            if i == 0:
                self.varList = ([],)
            else:
                self.varList = self.varList + ([],)

    def add_elements(self, *args):
        if len(args) == self.varNum:
            for i in range(len(args)):
                if not(args[i] is None):
                    self.varList[i].append(args[i])
                else:
                    self.varList[i].append(0)
        else:
            print('element number error')

    def add_info(self, info_dict):
        self.info_list.update(info_dict)

    def save2mat(self,save_path):
        time_temp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        save_fnt = save_path + time_temp + '.mat'
        data_dict = {'exptime': time_temp, }
        for i in range(self.varNum):
            data_dict.update({self.varNameList[i]: self.varList[i]})
        data_dict.update(self.info_list)
        sio.savemat(save_fnt, data_dict)
        print('Data saved: ' + save_fnt)

    def save2mat_tail(self, save_path, tail):
        time_temp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        save_fnt = save_path + time_temp + tail + '.mat'
        data_dict = {'exptime': time_temp, }
        for i in range(self.varNum):
            data_dict.update({self.varNameList[i]: self.varList[i]})
        data_dict.update(self.info_list)
        sio.savemat(save_fnt, data_dict)
        print('Data saved: ' + save_fnt)

class RealTimeSleeper:
    def __init__(self, sample_time):
        self._sample_time = sample_time
        self.loop_start_time = time.time()
        self.loop_flag = 0

    def init(self):
        self.loop_start_time = time.time()

    def sleep(self):
        self.loop_flag = self.loop_flag + 1
        current_time = time.time()

        loop_end_time = (self.loop_start_time + self._sample_time)
        sleep_time = loop_end_time - current_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print('warning: loop frequency lower than expected!')
        self.loop_start_time = time.time()

class Differentiator:
    def __init__(self, diff_steps=1):
        self.t_queue = deque([0] * diff_steps, maxlen=diff_steps)
        self.data_queue = deque([0] * diff_steps, maxlen=diff_steps)
        self.data_rate = 0

    def step(self, data_now, abstime):
        dt = abstime - self.t_queue[0]
        if dt == 0:
            self.t_queue.append(abstime)
        else:
            self.data_rate = (data_now - self.data_queue[0]) / dt
            self.t_queue.append(abstime)
            self.data_queue.append(data_now)

        return self.data_rate

class DataProcessor(object):
    def __init__(self, num_bodies, sample_rate):
        self.num_bodies = num_bodies
        self.sample_rate = sample_rate
        body_name = [i for i in range(1, self.num_bodies + 1)]
        self.keys = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        self.data_list = {name: {key: 0.0 for key in self.keys} for name in body_name}

        self.save_list_name = []
        self.save_list_data = []

        for body in body_name:
            for key in self.keys:
                self.save_list_name.append('b' + str(body) + '_' + key)
                self.save_list_data.append(0.0)

    def process_data(self, udp_data):
        for i in range(1, self.num_bodies + 1):
            x, y, z, qx, qy, qz, qw = struct.unpack("hhhhhhh", udp_data[(i*14 - 14):i*14])
            self.data_list[i]['x'] = x * 0.0005
            self.data_list[i]['y'] = y * 0.0005
            self.data_list[i]['z'] = z * 0.0005
            self.data_list[i]['qx'] = qx * 0.001
            self.data_list[i]['qy'] = qy * 0.001
            self.data_list[i]['qz'] = qz * 0.001
            self.data_list[i]['qw'] = qw * 0.001

        i = 0
        for body in self.data_list:
            for key in self.keys:
                self.save_list_data[i] = self.data_list[body][key]
                i = i + 1
        return self.data_list, self.save_list_data

class UdpRigidBodies(object):
    def __init__(self, udp_ip="0.0.0.0", udp_port=22222):
        self.len_data = 100
        self.udp_flag = 0
        self._udpStop = False
        self._udp_data = None
        self._udp_data_time = time.time()
        self._udpThread = None
        self._udpThread_on = False
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.num_bodies = 0

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self._sock.bind((self.udp_ip, self.udp_port))

        self.sample_rate = -1  # flag
        self.sample_rate = self.get_sample_rate()
        self.sample_time = 1 / self.sample_rate
        print('UDP receiver initialized')

    def get_sample_rate(self):
        if self.sample_rate == -1:
            print('Computing sample rate...')
            time_list = []
            for _ in tqdm(range(1000), desc="Processing...", leave=True, position=0):
                time_list.append(time.time())
                udp_data_temp, _ = self._sock.recvfrom(100)  # buffer size is 8192 bytes
                self._udp_data_time = time.time()
                self.len_data = len(udp_data_temp)
            d_time = np.diff(time_list)
            sample_time = np.mean(d_time)

            print('Sample rate: ', '%.2f' % (1/sample_time), 'Hz')
            print('UDP data size: ', '%.2f' % (self.len_data))

            return 1/sample_time
        else:
            return self.sample_rate

    def start_thread(self):
        if not self._udpThread_on:
            self._udpThread = threading.Thread(target=self._udp_worker, args=(), )
            self._udp_data = b'1'
            self._udpThread.start()
            self._udpThread_on = True
            time.sleep(1)
            print('Upd thread start')
            self.num_bodies = len(self._udp_data)/14
            if self.num_bodies % 1 == 0:
                self.num_bodies = int(self.num_bodies)
                print('Number of rigid bodies: ' + str(self.num_bodies))
            else:
                print('error: incorrect data')
        else:
            print('New upd thread is not started')

    def _udp_worker(self, ):
        if not self._udpThread_on:
            while not self._udpStop:
                self.udp_flag = self.udp_flag + 1
                udp_data_temp, _ = self._sock.recvfrom(self.len_data)  # buffer size is 8192 bytes
                self._udp_data_time = time.time()
                self._udp_data = udp_data_temp

    def stop_thread(self, ):
        # self._sync_on = False
        self._udpStop = True
        time.sleep(self.sample_time)
        print('upd thread stopped')

    def get_data(self, ):
        # get current data
        # self._sync_on = False
        # self._udp_data_ready.wait()
        return self._udp_data, self._udp_data_time

    def get_data_sync(self, ):
        self.udp_flag = self.udp_flag + 1
        udp_data_temp, _ = self._sock.recvfrom(self.len_data)  # buffer size is 8192 bytes
        self._udp_data = udp_data_temp
        return self._udp_data

class UdpRigidBodiesViCON(object):
    def __init__(self, udp_ip="0.0.0.0", udp_port=51001):
        self.len_data = 100
        self.udp_flag = 0
        self._udpStop = False
        self._udp_data = None
        self._udp_data_time = time.time()
        self._udpThread = None
        self._udpThread_on = False
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.num_bodies = 0

        self.block_size = 1024

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self._sock.bind((self.udp_ip, self.udp_port))

        self.sample_rate = -1  # flag
        self.sample_rate = self.get_sample_rate()
        self.sample_time = 1 / self.sample_rate
        print('UDP receiver initialized')
        
    def get_sample_rate(self):
        if self.sample_rate == -1:
            print('Computing sample rate...')
            time_list = []
            for _ in tqdm(range(1000), desc="Processing...", leave=True, position=0):
                time_list.append(time.time())
                udp_data_temp, _ = self._sock.recvfrom(self.block_size)  # buffer size is 8192 bytes
                self._udp_data_time = time.time()
                self.len_data = len(udp_data_temp)
            d_time = np.diff(time_list)
            sample_time = np.mean(d_time)

            print('Sample rate: ', '%.2f' % (1/sample_time), 'Hz')
            print('UDP data size: ', '%.2f' % (self.len_data))

            return 1/sample_time
        else:
            return self.sample_rate
        
    def start_thread(self):
        if not self._udpThread_on:
            self._udpThread = threading.Thread(target=self._udp_worker, args=(), )
            self._udpThread.start()
            self._udpThread_on = True
            # Wait until worker has received the first real packet
            print('Waiting for first packet...')
            while self._udp_data is None:
                time.sleep(0.01)
            print('Upd thread start')
            # Read num_bodies directly from Vicon packet header (bytes 4)
            self.num_bodies = struct.unpack_from('<B', self._udp_data, 4)[0]
            print('Number of rigid bodies: ' + str(self.num_bodies))
        else:
            print('New upd thread is not started')

    def _udp_worker(self, ):
        while not self._udpStop:
            self.udp_flag = self.udp_flag + 1
            udp_data_temp, _ = self._sock.recvfrom(self.block_size)
            self._udp_data_time = time.time()
            self._udp_data = udp_data_temp

    def stop_thread(self, ):
        self._udpStop = True
        time.sleep(self.sample_time)
        print('upd thread stopped')

    def get_data(self, ):
        # get current data
        # self._sync_on = False
        # self._udp_data_ready.wait()
        return self._udp_data, self._udp_data_time

    def get_data_sync(self, ):
        self.udp_flag = self.udp_flag + 1
        udp_data_temp, _ = self._sock.recvfrom(self.len_data)  # buffer size is 8192 bytes
        self._udp_data = udp_data_temp
        return self._udp_data
    
class DataProcessorViCON(object):
    """Parse ViCON Tracker UDP object stream (see Tracker UDP / Simulink docs).

    One datagram can contain **multiple** rigid bodies. Layout per packet:
    - Bytes 0–3: frame number (uint32 LE)
    - Byte 4: ``ItemsInBlock`` (number of object records in this datagram)
    - Then for each object, in order: ``ItemID`` (uint8), ``ItemDataSize`` (uint16 LE),
      24-byte null-padded name, six float64 (Trans X/Y/Z mm, Rot X/Y/Z rad).

    ``ItemID`` is 0 for standard object data for **every** object; objects are
    distinguished by **order** in the packet (first → ``b1``, second → ``b2``, …).
    """

    def __init__(self, num_bodies, sample_rate):
        self._ITEM_NAME_SIZE = 24
        self._ITEM_PAYLOAD =   struct.calcsize('<dddddd')  # 48 bytes (Trans + Rot)
        self._PER_OBJECT_HEADER = struct.calcsize('<BH')  # ItemID + ItemDataSize

        self.num_bodies = num_bodies
        self.sample_rate = sample_rate

        self.keys = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

        # Indexed 1 … num_bodies  (same convention as original)
        body_ids = list(range(1, self.num_bodies + 1))
        self.data_list = {b: {k: 0.0 for k in self.keys} for b in body_ids}

        # Flat list for logging / CSV — names like "b1_x", "b1_y", …
        self.save_list_name = []
        self.save_list_data = []
        for b in body_ids:
            for k in self.keys:
                self.save_list_name.append('b' + str(b) + '_' + k)
                self.save_list_data.append(0.0)

    def process_data(self, udp_data):
        if len(udp_data) < 5:
            return self.data_list, self.save_list_data

        # struct.unpack_from('<I', udp_data, 0)[0]  # frame_id — unused here
        items_in_block = struct.unpack_from('<B', udp_data, 4)[0]
        offset = 5

        for body_order in range(items_in_block):
            need = offset + self._PER_OBJECT_HEADER + self._ITEM_NAME_SIZE + self._ITEM_PAYLOAD
            if need > len(udp_data):
                break

            _item_id, _item_data_size = struct.unpack_from('<BH', udp_data, offset)
            offset += self._PER_OBJECT_HEADER

            offset += self._ITEM_NAME_SIZE
            x, y, z, RotX, RotY, RotZ = struct.unpack_from('<dddddd', udp_data, offset)
            offset += self._ITEM_PAYLOAD

            # 1-based index = order of object in this UDP packet (not ItemID)
            body_idx = body_order + 1
            if body_idx not in self.data_list:
                continue

            cx, sx = math.cos(RotX / 2), math.sin(RotX / 2)
            cy, sy = math.cos(RotY / 2), math.sin(RotY / 2)
            cz, sz = math.cos(RotZ / 2), math.sin(RotZ / 2)
            qw = cx * cy * cz - sx * sy * sz
            qx = sx * cy * cz + cx * sy * sz
            qy = cx * sy * cz - sx * cy * sz
            qz = cx * cy * sz + sx * sy * cz

            # Convert mm → metres  (ViCON transmits millimetres)
            self.data_list[body_idx]['x'] = x * 1e-3
            self.data_list[body_idx]['y'] = y * 1e-3
            self.data_list[body_idx]['z'] = z * 1e-3
            self.data_list[body_idx]['qx'] = qx
            self.data_list[body_idx]['qy'] = qy
            self.data_list[body_idx]['qz'] = qz
            self.data_list[body_idx]['qw'] = qw

        # Flatten into save_list_data
        i = 0
        for body in self.data_list:
            for key in self.keys:
                self.save_list_data[i] = self.data_list[body][key]
                i += 1

        return self.data_list, self.save_list_data

# --------------------------------------------------------------------------- #
# Interactive recorder = the original DataExchange logger above, with two
# additions only:
#   (1) it waits for SPACE to START (instead of recording immediately), and
#   (2) it also records the drone's video feed alongside the ViCON pose.
# The ViCON collection is the proven path unchanged — UdpRigidBodiesViCON +
# DataProcessorViCON + RealTimeSleeper + Differentiator + DataSaver at 100 Hz,
# same loop body (see _collect) — just gated by a record flag and saved with a
# shared timestamp so the video pairs with it. Press Q / ESC / Ctrl+C to STOP:
# pose + video stop together and are written to DataExchange/<stamp>.mat +
# <stamp>.mp4. The .mat is the same lab format the OLD yaw-maneuver pipeline
# already consumes:
#   python sync_log.py LOG.bbl DataExchange/<stamp>.mat --plot
#
# Camera-less (or --no-video): runs ViCON-only, SPACE is taken from the terminal,
# and only the .mat is written. Run with the cv2 venv:
#   ../.venv/bin/python UdpReceiver_datacollection.py
if __name__ == '__main__':
    import os
    import sys
    import argparse
    import datetime

    _HERE = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(
        description='The DataExchange ViCON logger, but SPACE-triggered and with '
                    'synced video. Saves <stamp>.mat (+ <stamp>.mp4) into '
                    'DataExchange/ for post-flight yaw-maneuver sync (sync_log.py).')
    # Camera defaults mirror drone_control/controller_v2/config.py (DEVICE_INDEX=4,
    # 1280x720 — must match camera_calibration.npz). The Cam Link can re-enumerate
    # 4<->5, so we auto-try the next index when the first fails to open.
    ap.add_argument('--device', type=int, default=4, help='camera /dev/videoN (default 4)')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--port', type=int, default=51001, help='ViCON UDP port (default 51001)')
    ap.add_argument('--no-video', action='store_true', help='ViCON-only, no camera')
    ap.add_argument('--out-dir', default=os.path.join(_HERE, 'DataExchange'),
                    help='where <stamp>.mat/.mp4 land (default DataExchange/)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- camera (same setup as the rest of the repo); optional --------------- #
    cv2 = None
    cap = None
    if not args.no_video:
        try:
            import cv2 as _cv2
            cv2 = _cv2
            for dev in (args.device, args.device + 1):
                c = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                if c.isOpened():
                    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    c.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                    c.set(cv2.CAP_PROP_FPS, 60)
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap = c
                    print(f'Camera: /dev/video{dev} {args.width}x{args.height}')
                    break
                c.release()
            if cap is None:
                print('Camera open FAILED — running ViCON-only (no video).')
        except Exception as e:  # noqa: BLE001 — missing cv2 just disables video
            print(f'cv2 unavailable ({e}) — running ViCON-only (no video).')
    HAS_VIDEO = cap is not None

    # ---- ViCON receiver + processors: the original setup, unchanged ---------- #
    UDP = UdpRigidBodiesViCON(udp_port=args.port)   # measures sample rate, as before
    UDP.start_thread()
    DP = DataProcessorViCON(UDP.num_bodies, UDP.sample_rate)
    print('Save list names:', DP.save_list_name)

    RTS = RealTimeSleeper(0.01)                     # 100 Hz loop, same as original
    Diff_X = Differentiator(diff_steps=2)
    Diff_Y = Differentiator(diff_steps=2)
    Diff_Z = Differentiator(diff_steps=2)
    saver = DataSaver('Abs_time', *tuple(DP.save_list_name),
                      'b1_x_dot', 'b1_y_dot', 'b1_z_dot')
    rigid_body_index = 1

    rec = {'on': False, 'stamp': None, 't0': None}

    # ---- video writer thread: owns the camera, writes only while recording --- #
    frame_lock = threading.Lock()
    latest = {'frame': None}
    writer_holder = {'w': None}
    video_stop = threading.Event()

    def _video_worker():
        while not video_stop.is_set():
            ok, frame = cap.read()
            if not ok:
                continue
            with frame_lock:
                latest['frame'] = frame
            if rec['on'] and writer_holder['w'] is not None:
                writer_holder['w'].write(frame)

    if HAS_VIDEO:
        threading.Thread(target=_video_worker, daemon=True).start()

    # ---- the original loop body, factored out (unchanged behavior) ----------- #
    def _collect():
        Abs_time = RTS.loop_start_time - rec['t0']
        data_raw, udp_time = UDP.get_data()
        data, save_list_data = DP.process_data(data_raw)
        Diff_X.step(data[rigid_body_index]['x'], udp_time)
        Diff_Y.step(data[rigid_body_index]['y'], udp_time)
        Diff_Z.step(data[rigid_body_index]['z'], udp_time)
        saver.add_elements(Abs_time, *tuple(save_list_data),
                           Diff_X.data_rate, Diff_Y.data_rate, Diff_Z.data_rate)

    def _arm():
        rec['stamp'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if HAS_VIDEO:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 1 else 30.0
            writer_holder['w'] = cv2.VideoWriter(
                os.path.join(args.out_dir, rec['stamp'] + '.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'), fps, (args.width, args.height))
        RTS.init()
        rec['t0'] = time.time()
        rec['on'] = True
        print(f"REC {rec['stamp']} — ViCON{' + video' if HAS_VIDEO else ''} (t=0). "
              f"{'Q/ESC' if HAS_VIDEO else 'SPACE/ENTER'} or Ctrl+C to stop & save.")

    def _save():
        # Same dict shape as DataSaver.save2mat (exptime + each var list), but with
        # our shared stamp so the .mat pairs with the .mp4.
        if not saver.varList[0]:
            print('No ViCON samples recorded — nothing saved.')
            return
        data_dict = {'exptime': rec['stamp']}
        for nm, lst in zip(saver.varNameList, saver.varList):
            data_dict[nm] = lst
        if HAS_VIDEO:
            data_dict['video_file'] = rec['stamp'] + '.mp4'
        mat_path = os.path.join(args.out_dir, rec['stamp'] + '.mat')
        sio.savemat(mat_path, data_dict)
        n = len(saver.varList[0])
        print(f'\nSaved ViCON log: {mat_path}  ({n} samples)')
        if HAS_VIDEO:
            print(f'Saved video:     {os.path.join(args.out_dir, rec["stamp"] + ".mp4")}')
        print('Sync the blackbox in post (yaw maneuver):\n'
              f'  python sync_log.py LOG.bbl {mat_path} --plot')

    try:
        if HAS_VIDEO:
            WIN = 'drone feed  [SPACE]=start  [Q]=stop & save'
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            last_show = 0.0
            while True:
                if rec['on']:
                    _collect()
                now = time.time()
                if now - last_show > 0.033:          # ~30 Hz display, Vicon stays 100 Hz
                    with frame_lock:
                        frame = latest['frame']
                    if frame is not None:
                        disp = frame.copy()
                        if rec['on']:
                            cv2.putText(disp, f"REC {now-rec['t0']:5.1f}s   "
                                        f"vicon {len(saver.varList[0])}", (12, 34),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        else:
                            cv2.putText(disp, "SPACE = start    Q = quit", (12, 34),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                        cv2.imshow(WIN, disp)
                    last_show = now
                k = cv2.waitKey(1) & 0xFF
                if k == ord(' ') and not rec['on']:
                    _arm()
                elif k in (ord('q'), 27):
                    break
                if rec['on']:
                    RTS.sleep()
            cv2.destroyAllWindows()

        elif sys.stdin.isatty():
            import termios
            import tty
            import select
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                print('Press SPACE to start ViCON recording (Q to quit)...')
                while not rec['on']:
                    if select.select([sys.stdin], [], [], 0.2)[0]:
                        ch = sys.stdin.read(1)
                        if ch == ' ':
                            _arm()
                        elif ch in ('q', '\x03'):
                            raise KeyboardInterrupt
                while True:
                    _collect()
                    if select.select([sys.stdin], [], [], 0)[0]:
                        if sys.stdin.read(1) in (' ', '\r', '\n', 'q', '\x03'):
                            break
                    RTS.sleep()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        else:
            input('Press ENTER to start ViCON recording...')
            _arm()
            while True:                               # stop with Ctrl+C
                _collect()
                RTS.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        rec['on'] = False
        video_stop.set()
        if writer_holder['w'] is not None:
            writer_holder['w'].release()
        if cap is not None:
            cap.release()
        UDP.stop_thread()
        _save()
