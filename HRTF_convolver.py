"""Implementation of a 3D Audio Panner using the CIPIC HRTF Database.

Usage:

Select a subject from the CIPIC database. You should select a subject 
with similar anthropometric measurements as yourself for the best
experience.

    - Note: Due to storage limitations, the repository has only 4
            subjects of the database to choose from. The full database
            is ~170MB and has 45 subjects. It can be downloaded for free
            at:

            https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/
            
            In order to make it work, you should simply replace the 
            folder ´CIPIC_hrtf_database´ with the one you downloaded.

Press 'Play' to start playing the default sound file. Move the Azimuth
and Elevation sliders to position the sound in the 3D space.

You can load your own audio file in File/Load audio file. Also, there
are other sound samples in the folder resources/sound.

*IMPORTANT:* For now, the only working format is a mono WAV file at
44100 Hz sample rate and 16 bit depth. 

You can save the file at the specified pair of Azimuth/Elevation in
File/Save audio file.

Lastly, you can choose to use a crossover in order not to spatialize low 
frequencies, since low frequencies are non-directional in nature. Go to 
Settings/Change cutoff frequency to set the desired frequency. By
default, crossover is set at 200 Hz.

Author:         Francisco Rotea
                (Buenos Aires, Argentina)
Repository:     https://github.com/franciscorotea
Email:          francisco.rotea@gmail.com
Version:        0.1
Last Revised:   Thursday, June 04, 2020

"""

import os
import wave
import itertools

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import pyaudio

import scipy.io
from scipy.io import wavfile
import scipy.spatial
import scipy.signal

import numpy as np
import numpy.linalg

from PIL import ImageTk

# Values of azimuth and elevation angles measured in the CIPIC database. 
# See ´CIPIC_hrtf_database/doc/hrir_data_documentation.pdf´ for
# information about the coordinate system and measurement procedure.

AZIMUTH_ANGLES = [
    -80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15,
    20, 25, 30, 35, 40, 45, 55, 65, 80,
    ]

ELEVATION_ANGLES = -45 + 5.625*np.arange(0, 50)

POINTS = np.array(list(itertools.product(AZIMUTH_ANGLES, ELEVATION_ANGLES)))

# Get indexes from angles.

AZ = dict(zip(AZIMUTH_ANGLES, np.arange(len(AZIMUTH_ANGLES))))
EL = dict(zip(ELEVATION_ANGLES, np.arange(len(ELEVATION_ANGLES))))

# Load anthropometric measurements data from the CIPIC database.

# See ´CIPIC_hrtf_database/doc/anthropometry.pdf´ for information about 
# the parameters definition.

anthro_data = scipy.io.loadmat('CIPIC_hrtf_database/anthropometry/anthro.mat')

PARAMETERS = {
    'info': ['Age:', 'Sex:', 'Weight:'],
    'X': ['Head width:', 'Head height:', 'Head depth:', 'Pinna offset down:', 
          'Pinna offset back:', 'Neck width:', 'Neck height:', 'Neck depth:', 
          'Torso top width:', 'Torso top height:', 'Torso top depth:', 
          'Shoulder width:', 'Head offset forward:', 'Height:',
          'Seated height:', 'Head circumference:', 'Shoulder circumference:'],
    'D': ['Cavum concha height', 'Cymba concha height', 'Cavum concha width', 
          'Fossa height', 'Pinna height', 'Pinna width', 
          'Intertragal incisure width', 'Cavum concha depth'],
    'theta': ['Pinna rotation angle:', 'Pinna flare angle:']
}

L_R = [' (left):', ' (right):']   # To use with 'D' and 'theta' parameters.

# Clean anthropometric data for display.

for key, value in anthro_data.items():
    if key not in ['__header__', '__version__', '__globals__', 'id', 'sex']:
        if key == 'age':
            anthro_data[key][np.isnan(anthro_data[key])] = 0
            anthro_data[key] = np.squeeze(value.astype('int')).astype('str')
            anthro_data[key][anthro_data[key] == '0'] = '-'
        else:
            anthro_data[key] = np.around(np.squeeze(value), 1).astype('str')
            anthro_data[key][anthro_data[key] == 'nan'] = '-'

# Get indexes from ID's.

ANTHRO_ID = anthro_data['id'].tostring()
ID_TO_IDX = dict(zip(ANTHRO_ID, range(len(ANTHRO_ID))))

# Generate a list with all subject's ID present in the database.

FOLDERS = os.listdir('CIPIC_hrtf_database/standard_hrir_database')
SUBJECT_ID = [id_.strip('subject_') for id_ in FOLDERS if id_ != 'show_data']

# Initialization variables for audio stream.

SAMPLE_RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 2

# Initialization variables for overlap-save algorithm.

# L = Window size.
# M = Length of impulse response.
# N = Size of the DFT. Since the length of the convolved signal will be
#     L+M-1, it is rounded to the nearest power of 2 for efficient fft
#     calculation.

L = 2048
M = 200
N = int(2**np.ceil(np.log2(np.abs(L+M-1))))

L = N - M + 1

# Preallocate interpolated impulse responses.

interp_hrir_l = np.zeros(M)
interp_hrir_r = np.zeros(M)

# Radius of the azimuth and elevation circles (RADIUS_1) and radius of
# the circular trajectory in which they move (RADIUS_2). Both in pixels.

RADIUS_1 = 10
RADIUS_2 = 130

# Filepath for the head images.

FRONT_IMG_PATH = 'resources/images/head_img_front.png'
SIDE_IMG_PATH = 'resources/images/head_img_side.png'

# From number of channels to string (for display).

S_M = {1: 'Mono', 2: 'Stereo'}

# Width and height of windows, in pixels.

MAIN_PAGE_SIZE = (870, 370)
START_PAGE_SIZE = (870, 460)

def butter_lp(cutoff, fs, order):
    """Design of a digital Butterworth low pass filter with a
    second-order section format for numerical stability."""

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    sos = scipy.signal.butter(N=order,
                              Wn=normal_cutoff,
                              btype='lowpass',
                              output='sos')

    return sos


def butter_lp_filter(signal, cutoff, fs=SAMPLE_RATE, order=1):
    """Filter a signal with the filter designed in ´butter_lp´.

    Filfilt applies the linear filter twice, once forward and once
    backwards, so that he combined filter has zero phase delay."""

    sos = butter_lp(cutoff=cutoff, fs=fs, order=order)
    out = scipy.signal.sosfiltfilt(sos, signal)

    return out


def butter_hp(cutoff, fs, order):
    """Design of a digital Butterworth high pass filter with a
    second-order section format for numerical stability."""

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    sos = scipy.signal.butter(N=order,
                              Wn=normal_cutoff,
                              btype='highpass',
                              output='sos')

    return sos


def butter_hp_filter(signal, cutoff, fs=SAMPLE_RATE, order=1):
    """Filter a signal with the filter designed in ´butter_hp´.

    Filfilt applies the linear filter twice, once forward and once
    backwards, so that he combined filter has zero phase delay."""

    sos = butter_hp(cutoff=cutoff, fs=fs, order=order)
    out = scipy.signal.sosfiltfilt(sos, signal)

    return out


def create_triangulation(points):
    """Generate a triangular mesh from HRTF measurement points (azimuth,
    elevation) using the Delaunay triangulation algorithm."""

    triangulation = scipy.spatial.Delaunay(points)

    return triangulation


def calculate_T_inv(triang, points):
    """Performs the calculation of the inverse of matrix T for all
    triangles in the triangulation and stores it in an array.

    Matrix T is defined as:

        T = [[A - C],
             [B - C]]

    where A, B and C are vertices of the triangle.

    Since T is independent of source position X, the precalculation of T
    allows to reduce the operational counts for finding the
    interpolation weights.
    
    For a more comprehensive explanation of this procedure, refer to:

    Gamper, H., Head-related transfer function interpolation in azimuth,
    elevation, and distance, J. Acoust. Soc. Am. 134 (6), December 2013.

    """

    A = points[triang.simplices][:,0,:]
    B = points[triang.simplices][:,1,:]
    C = points[triang.simplices][:,2,:]

    T = np.empty((2*A.shape[0], A.shape[1]))

    T[::2,:] = A - C
    T[1::2,:] = B - C

    T = T.reshape(-1, 2, 2)

    T_inv = np.linalg.inv(T)

    return T_inv


def interp_hrir(triang, points, T_inv, hrir_l, hrir_r, azimuth, elevation):
    """Estimate a HRTF for any point X lying inside the triangular mesh 
    calculated.

    This is done by interpolating the vertices of the triangle enclosing
    X. Given a triangle with vertices A, B and C, any point X inside the
    triangle can be represented as a linear combination of the vertices:

    X = g_1 * A + g_2 * B + g_3 * C

    where g_i are scalar weights. If the sum of the weights is equal to
    1, these are barycentric coordinates of point X. Given a desired
    source position X, barycentric interpolation weights are calculated
    as:

    [g_1, g_2] = (X - C) * T_inv
    g_ 3 = 1 - g_1 - g_2

    Barycentric coordinates are used as interpolation weights for
    estimating the HRTF at point X as the weighted sum of the HRTFs
    measured at A, B and C, respectively.

    One of the main advantages of this interpolation approach is that
    it does not cause discontinuities in the interpolated HRTFs: for a
    source moving smoothly from one triangle to another, the HRTF
    estimate changes smoothly, even at the crossing point.

    For a more comprehensive explanation of the interpolation algorithm,
    please refer to:

    Gamper, H., Head-related transfer function interpolation in azimuth,
    elevation, and distance, J. Acoust. Soc. Am. 134 (6), December 2013.

    """

    position = [azimuth, elevation]
    triangle = triang.find_simplex(position)
    vert = points[triang.simplices[triangle]]

    X = position - vert[2]
    g = np.dot(X, T_inv[triangle])

    g_1 = g[0]
    g_2 = g[1]
    g_3 = 1 - g_1 - g_2

    if g_1 >= 0 and g_2 >= 0 and g_3 >= 0:
        interp_hrir_l[:] = g_1 * hrir_l[AZ[vert[0][0]]][EL[vert[0][1]]][:] + \
                           g_2 * hrir_l[AZ[vert[1][0]]][EL[vert[1][1]]][:] + \
                           g_3 * hrir_l[AZ[vert[2][0]]][EL[vert[2][1]]][:]

        interp_hrir_r[:] = g_1 * hrir_r[AZ[vert[0][0]]][EL[vert[0][1]]][:] + \
                           g_2 * hrir_r[AZ[vert[1][0]]][EL[vert[1][1]]][:] + \
                           g_3 * hrir_r[AZ[vert[2][0]]][EL[vert[2][1]]][:]
    
    return interp_hrir_l, interp_hrir_r


def get_circle_coords(angle, offset_x, offset_y):
    """Generate coordinates to create and move a circle in tkinter.

    (x0, y0): top left corner.
    (x1, y1): bottom right corner.

    """

    x_center = RADIUS_2*np.cos(angle) + offset_x
    y_center = RADIUS_2*np.sin(angle) + offset_y

    x0 = x_center - RADIUS_1
    y0 = y_center - RADIUS_1
    x1 = x_center + RADIUS_1
    y1 = y_center + RADIUS_1

    return (x0, y0, x1, y1)


TRI = create_triangulation(points=POINTS)
T_INV = calculate_T_inv(triang=TRI, points=POINTS)


class App(tk.Tk):
    """Controller object, this is the common point of interaction for 
    the two pages (start and main)."""

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.title(self, '3D Audio Panner')
        tk.Tk.iconbitmap(self, default='resources/images/icon.ico')

        # Select a subject from the database by default.

        self.subject = '003'

        # Create a container in which frames are stuck on top of each
        # other. To make one visible, it is raised above using
        # ´show_frame´.

        container = tk.Frame(self, borderwidth=5)

        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        
        for F in (StartPage, MainPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame('StartPage')

    def show_frame(self, page_name):
        """Show a frame for the given page name and adjust geometries"""

        frame = self.frames[page_name]
        frame.tkraise()

        if page_name == 'MainPage':
            menu_bar = frame.menu_bar(self)
            self.configure(menu=menu_bar)
            self.geometry(f'{MAIN_PAGE_SIZE[0]}x{MAIN_PAGE_SIZE[1]}')
            self.minsize(width=MAIN_PAGE_SIZE[0], height=MAIN_PAGE_SIZE[1])
        else:
            self.geometry(f'{START_PAGE_SIZE[0]}x{START_PAGE_SIZE[1]}')
            self.minsize(width=START_PAGE_SIZE[0], height=START_PAGE_SIZE[1])

class StartPage(tk.Frame):
    """Page to select a subject from the CIPIC database."""

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        select_label = tk.Label(self,
                                text='Select a subject from the database:')

        select_label.grid(row=0, column=0, columnspan=4)

        self.labelframes = [tk.LabelFrame(self, text='General'),
                            tk.LabelFrame(self, text='Head and Torso'),
                            tk.LabelFrame(self, text='Pinna'),
                            tk.LabelFrame(self, text='Pinna angles')]

        for i, labelframe in enumerate(self.labelframes):
            labelframe.grid(row=2, column=i, sticky='nsew')

        for i, (param, values) in enumerate(PARAMETERS.items()):
            if param == 'D' or param == 'theta':
                for j, value in enumerate(itertools.product(L_R, values)):
                    label = tk.Label(self.labelframes[i],
                                     text=value[1] + value[0])
                    label.grid(row=j, column=0, sticky='ew')
            else:
                for j, value in enumerate(values):
                    label = tk.Label(self.labelframes[i], text = value)
                    label.grid(row=j, column=0, sticky='ew')

        self.selected_subject = tk.StringVar(self)
        self.selected_subject.trace('w', self.display_anthro_data)

        subject_popup = ttk.OptionMenu(self,
                                       self.selected_subject,
                                       SUBJECT_ID[0],
                                       *SUBJECT_ID)

        subject_popup.grid(row=1, column=0, columnspan=4, sticky='nsew')

        select_button = ttk.Button(self,
                                   text='Select',
                                   command=self.select_subject)

        select_button.grid(row=3, column=0, columnspan=4, sticky='nsew')

        for row in range(self.grid_size()[1]):
            self.rowconfigure(row, weight=1)

        for column in range(self.grid_size()[0]):
            self.columnconfigure(column, weight=1)

    def display_anthro_data(self, *args):
        """Display the anthropometric data for the subject selected."""

        idx = ID_TO_IDX[int(self.selected_subject.get())]

        # Don't display units if there is no data available.

        if anthro_data['WeightKilograms'][idx] == '-':
            units = ''
        else:
            units = ' Kg'

        for i, param in enumerate(['age', 'sex', 'WeightKilograms']):
            if param == 'WeightKilograms':
                label = tk.Label(master=self.labelframes[0],
                                 text=anthro_data[param][idx] + units)
            else:
                label = tk.Label(master=self.labelframes[0],
                                 text=anthro_data[param][idx])
            label.grid(row=i, column=1, sticky='ew')

        for i, (param, values) in enumerate(PARAMETERS.items()):

            if param != 'info':

                # Don't display units if there is no data available.

                if anthro_data[param][idx][i] == '-':
                    units = ''
                else:
                    if param == 'theta':
                        units = '°'
                    else:
                        units = ' cm'

                # Adjust for parameters with left and right values.

                LR_const = 2 if param == 'D' or param == 'theta' else 1

                for j in range(LR_const * len(values)):
                    label = tk.Label(master=self.labelframes[i],
                                     text=anthro_data[param][idx][j] + units)
                    
                    label.grid(row=j, column=1, sticky='ew')

        for labelframe in self.labelframes:
           for column in range(labelframe.grid_size()[0]):
                labelframe.columnconfigure(column, weight=1)

    def select_subject(self):
        """Callback function for the Select button. Stores the selected
        subject in a variable, used to load the HRIR data."""

        self.controller.subject = self.selected_subject.get()
        self.controller.show_frame('MainPage')

        print(f'Selected subject_{self.controller.subject} from the database.')

class MainPage(tk.Frame):
    """Main page of the application."""

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        # Load head images.

        self.head_img_front = ImageTk.PhotoImage(file=FRONT_IMG_PATH)
        self.head_img_side = ImageTk.PhotoImage(file=SIDE_IMG_PATH)

        # Initialize streaming-related variables.

        self.play_mode = False
        self.finished = False
        
        self.p = pyaudio.PyAudio()

        # By default, use 200 Hz as crossover frequency.

        self.crossover = tk.BooleanVar(self)
        self.crossover.set(True)

        self.cutoff_freq = 200

        # By default, load an audio file from the ´sounds´ folder.

        self.audio_path = 'resources/sounds/snare_roll.wav'
        self.audio = wave.open(self.audio_path, 'rb')

        # Load HRIR data for the selected subject.

        hrir = scipy.io.loadmat('CIPIC_hrtf_database/standard_hrir_database/'
                                'subject_' + self.controller.subject +
                                '/hrir_final.mat')

        self.hrir_l = np.array(hrir['hrir_l'])
        self.hrir_r = np.array(hrir['hrir_r'])

        # Definition of Azimuth-related widgets.

        self.canvas_az = tk.Canvas(self)
        self.canvas_az.bind("<Configure>",
                            lambda event: self.resize_callback(event,
                                                               circle_az,
                                                               circle_el))

        self.canvas_az.grid(row=0, column=0, sticky='nsew')

        self.img_front = self.canvas_az.create_image(0,
                                                     0,
                                                    image=self.head_img_front)

        coords_az = get_circle_coords(angle=3*np.pi/2,
                                      offset_x=self.canvas_az.winfo_width()/2,
                                      offset_y=self.canvas_az.winfo_height()/2)

        circle_az = self.canvas_az.create_oval(coords_az, fill='#d7191c')

        self.azimuth = tk.DoubleVar(self)
        
        slider_az = tk.Scale(self,
                             variable=self.azimuth,
                             orient=tk.HORIZONTAL,
                             label='Azimuth',
                             from_=min(AZIMUTH_ANGLES),
                             to=max(AZIMUTH_ANGLES),
                             resolution=0.1,
                             sliderlength=50,
                             command=lambda _: self.move_circle_az(circle_az))
        
        slider_az.grid(row=1, column=0, columnspan=2, sticky='ew')

        # Definition of Elevation-related widgets.

        self.canvas_el = tk.Canvas(self)
        self.canvas_el.bind("<Configure>",
                            lambda event: self.resize_callback(event,
                                                               circle_az,
                                                               circle_el))

        self.canvas_el.grid(row=0, column=1, sticky='nsew')

        self.img_side = self.canvas_el.create_image(0,
                                                    0,
                                                    image=self.head_img_side)

        coords_el = get_circle_coords(angle=3*np.pi/2,
                                      offset_x=self.canvas_el.winfo_width()/2,
                                      offset_y=self.canvas_el.winfo_height()/2)

        circle_el = self.canvas_el.create_oval(coords_el, fill='#2b83ba')

        self.elevation = tk.DoubleVar(self)

        slider_el = tk.Scale(self,
                             variable=self.elevation,
                             orient=tk.VERTICAL,
                             label='Elevation',
                             from_=min(ELEVATION_ANGLES),
                             to=max(ELEVATION_ANGLES),
                             resolution=0.1,
                             sliderlength=50,
                             command=lambda _: self.move_circle_el(circle_el))
        
        slider_el.grid(row=0, column=2, sticky='ns')

        self.play_button = tk.Button(self,
                                     text='Play',
                                     command=self.play_audio)

        self.play_button.grid(row=1, column=2, sticky='nsew')

        for r in range(self.grid_size()[1]):
            self.rowconfigure(r, weight=1)
        for c in range(self.grid_size()[0]):
            self.columnconfigure(c, weight=1)

    def menu_bar(self, root):
        """Create a menu bar to be displayed in the Main page."""
        
        menu = tk.Menu(root)

        file = tk.Menu(menu, tearoff=0)

        file.add_command(label='Load audio file',
                         command=self.load_audio_dialog)
        file.add_command(label='Save audio file',
                         command=self.save_audio_dialog)

        file.add_separator()

        file.add_command(label='Exit', command=root.destroy)

        menu.add_cascade(label='File', menu=file)

        settings = tk.Menu(menu, tearoff=0)

        settings.add_command(label='Change cutoff frequency',
                             command=self.change_cutoff_freq)
        settings.add_command(label='Change subject',
                             command=lambda: self.controller.show_frame('Start'
                                                                        'Page')
                            )

        menu.add_cascade(label='Settings', menu=settings)

        help_ = tk.Menu(menu, tearoff = 0)

        help_.add_command(label='About', command=self.about)

        menu.add_cascade(label='Help', menu=help_)

        return menu

    def about(self):
        """Callback function for 'About' option in Help menu."""

        win = tk.Toplevel(self)
        win.title('About')
        win.resizable(False, False)
        label = tk.Label(win, text='*** 3D Audio Panner - v0.1 ***\n\n'
                                   'Written in Python 3\nby\n'
                                   'Francisco Rotea\n\n'
                                   '2020')
        label.pack()

    def change_cutoff_freq(self):
        """Callback function for 'Change cutoff frequency' option in Settings
        menu."""

        win = tk.Toplevel(self)
        win.title('Cutoff frequency')
        win.resizable(False, False)

        def disable_spinbox(*args):
            if crossover.get():
                spbox['state'] = 'normal'
            else:
                spbox['state'] = 'disabled'

        def save_settings():
            self.crossover.set(crossover.get())
            self.cutoff_freq = float(spbox.get())
            win.destroy()

        crossover = tk.BooleanVar(win)
        crossover.set(self.crossover.get())
        crossover.trace('w', disable_spinbox)

        checkbox = tk.Checkbutton(win,
                                  text='Crossover',
                                  variable=crossover)

        checkbox.grid(row=0, column=0, columnspan=2, sticky='nsew')

        start_value = tk.StringVar(win)
        start_value.set(str(self.cutoff_freq))

        spbox = tk.Spinbox(win,
                           from_=50,
                           to=500,
                           textvariable=start_value,
                           state='normal' if crossover.get() else 'disabled')

        spbox.grid(row=1, column=0, sticky='nsew')

        hz_label = tk.Label(win, text='Hz')
        hz_label.grid(row=1, column=1, sticky='nsew')

        ok_button = ttk.Button(win, text='OK', command=save_settings)
        ok_button.grid(row=2, column=1, sticky='nsew')

        cancel_button = ttk.Button(win, text='Cancel', command=win.destroy)
        cancel_button.grid(row=2, column=0, sticky='nsew')

    def load_audio_dialog(self):
        """Callback function for 'Load audio file' option in File menu."""

        self.audio_path = filedialog.askopenfilename(initialdir='resources/'
                                                                'sounds',
                                                     title='Select wav file',
                                                     filetypes=[('Audio file',
                                                                 '*.wav')])
        if self.audio_path:
            try:
                audio = wave.open(self.audio_path, 'rb')
                if (audio.getnchannels() != 1 
                        or audio.getsampwidth() != 2 
                        or audio.getframerate() != 44100):
                    messagebox.showerror('Error',
                                         'Loaded file is '
                                         f'{S_M[audio.getnchannels()]} at '
                                         f'{audio.getframerate()} Hz / '
                                         f'{audio.getsampwidth() * 8} bits.'
                                         '\nWav must be Mono at 44100 Hz / 16'
                                         ' bits.')
                else:
                    self.audio = audio
            except wave.Error:
                messagebox.showerror('Error', 'Your file could not be loaded.'
                                              '\nTry changing bitrate from 32'
                                              ' bits to 16 bits.')

    def save_audio_dialog(self):
        """Callback function for 'Save audio file' option in File menu."""

        path = filedialog.asksaveasfilename(initialdir='resources/saved',
                                            title='Save audio file',
                                            filetypes=[('Audio file', '*.wav')]
                                            )

        hrir_l, hrir_r = interp_hrir(triang=TRI,
                                     points=POINTS,
                                     T_inv=T_INV,
                                     hrir_l=self.hrir_l,
                                     hrir_r=self.hrir_r,
                                     azimuth=self.azimuth.get(),
                                     elevation=self.elevation.get())

        _, data = scipy.io.wavfile.read(self.audio_path)

        if self.crossover.get():
            data_lp = butter_lp_filter(signal=data,
                                       cutoff=self.cutoff_freq)
            data_hp = butter_hp_filter(signal=data,
                                       cutoff=self.cutoff_freq)
        else:
            data_lp = np.zeros_like(data)
            data_hp = data

        l_channel = np.convolve(data_hp, hrir_l) + np.pad(array=data_lp,
                                                          pad_width=(0, M-1))

        r_channel = np.convolve(data_hp, hrir_r) + np.pad(array=data_lp,
                                                          pad_width=(0, M-1))

        out = np.vstack((l_channel, r_channel)).T

        if path:
            try:
                scipy.io.wavfile.write(filename=path + '.wav',
                                       rate=SAMPLE_RATE,
                                       data=np.asarray(out, dtype=np.int16))

                messagebox.showinfo('Information', 'File saved successfully.')
            except:
                messagebox.showerror('Error', 'Your file could not be saved.')

    def resize_callback(self, event, circle_az, circle_el):
        """Keep widgets in canvas centered when resizing the window."""

        self.canvas_az.coords(self.img_front, event.width/2, event.height/2)
        self.canvas_el.coords(self.img_side, event.width/2, event.height/2)

        az_coords = get_circle_coords(angle=(self.azimuth.get()*np.pi/180
                                             + 3*np.pi/2),
                                      offset_x=event.width/2,
                                      offset_y=event.height/2)

        self.canvas_az.coords(circle_az, az_coords)

        el_coords = get_circle_coords(angle=-self.elevation.get()*np.pi/180,
                                      offset_x=event.width/2,
                                      offset_y=event.height/2)
        
        self.canvas_el.coords(circle_el, el_coords)

        self.canvas_width = event.width
        self.canvas_height = event.height

    def move_circle_az(self, circle):
        """Callback function for Azimuth slider."""

        az_coords = get_circle_coords(angle=(self.azimuth.get()*np.pi/180
                                             + 3*np.pi/2),
                                      offset_x=self.canvas_width/2,
                                      offset_y=self.canvas_height/2)

        self.canvas_az.coords(circle, az_coords)

    def move_circle_el(self, circle):
        """Callback function for Elevation slider."""

        el_coords = get_circle_coords(angle=-self.elevation.get()*np.pi/180,
                                      offset_x=self.canvas_width/2,
                                      offset_y=self.canvas_height/2)

        self.canvas_el.coords(circle, el_coords)

    def init_stream(self): 
        """Initialize the audio stream in callback mode and performs
        real-time convolution between the audio signal and the HRIR.

        The callback function is called in a separate thread whenever
        there's new audio data to play. This callback function performs
        the convolution between the audio signal and the interpolated
        HRIR at the specified position. In order to perform convolution
        in real time, the overlap-save method is implemented. This
        method consists on breaking the input audio signal into chunks
        of size L, transform the chunks into the frequency domain with
        the FFT and multiply it by the impulse response's DFT (i.e.
        convolution in time domain), transform back to the time domain
        and lop on the last L samples from the resulting L+M-1 chunk.

        For a detailed explanation of the algorithm, please refer to:

        Oppenheim, A. V. and Schafer, R. W., Discrete-Time Signal
        Processing, Second Edition, Prentice Hall, Chapter 8,
        pp. 582-588.

        """

        # Buffer to save overlap in each iteration (last M-1 samples are
        # appended to the start of each block).

        buffer_OLAP = np.zeros(M - 1)

        def callback(in_data, frame_count, time_info, status):
            if status:
                print(f'Playback Error: {status}')

            # Load HRIR data for the selected subject.

            hrir = scipy.io.loadmat('CIPIC_hrtf_database/'
                                    '/standard_hrir_database/subject_'
                                    + self.controller.subject +
                                    '/hrir_final.mat')

            self.hrir_l = np.array(hrir['hrir_l'])
            self.hrir_r = np.array(hrir['hrir_r'])

            # Interpolate to get the HRIR at the position selected.

            hrir_l, hrir_r = interp_hrir(triang=TRI,
                                         points=POINTS,
                                         T_inv=T_INV,
                                         hrir_l=self.hrir_l,
                                         hrir_r=self.hrir_r,
                                         azimuth=self.azimuth.get(),
                                         elevation=self.elevation.get())

            # Compute DFT of impulse response and zero-pad to match the
            # length of the blocks.

            h = np.vstack(([hrir_l, hrir_r]))
            h = np.hstack((h, np.zeros((2, N-(M-1)))))

            H = np.fft.fft(h, N)

            # Read audio chunk, transform to numpy array, and prepend
            # the saved overlap to the block. Then, pad with zeros at
            # the end to make up for uneven chunk of samples at the end
            # of the signal.

            data = self.audio.readframes(frame_count)

            x_r = np.frombuffer(data, dtype=np.int16)
            x_r_overlap = np.hstack((buffer_OLAP, x_r))
            x_r_zeropad = np.hstack((x_r_overlap,
                                     np.zeros(N - len(x_r_overlap))))

            # The last M-1 samples are saved for the next iteration.

            buffer_OLAP[:] = x_r_zeropad[N-(M-1):N]

            # Filter the signal so that low frequencies are not
            # spatialized.

            if self.crossover.get():
                x_r_zeropad_lp = butter_lp_filter(signal=x_r_zeropad, 
                                                  cutoff=self.cutoff_freq)
                x_r_zeropad_hp = butter_hp_filter(signal=x_r_zeropad,
                                                  cutoff=self.cutoff_freq)
            else:
                x_r_zeropad_lp = np.zeros_like(x_r_zeropad)
                x_r_zeropad_hp = x_r_zeropad

            # Get DFT of the block and perform the convolution for both
            # channels (left and right).

            Xm = np.tile(np.fft.fft(x_r_zeropad_hp, N), (2, 1))
            Ym = Xm * H

            ym = np.real(np.fft.ifft(Ym))

            # Combine the results into a stereo signal (interleaved).

            out_data = np.empty((ym[:, M-2:N].size), dtype=np.int16)

            out_data[0::2] = ym[0, M-2:N] + x_r_zeropad_lp[M-2:N]
            out_data[1::2] = ym[1, M-2:N] + x_r_zeropad_lp[M-2:N]

            if data:
                self.finished = False
                return (out_data, pyaudio.paContinue)
            else:
                self.finished = True
                self.play_button.config(text='Play')
                self.play_mode = False
                self.audio.rewind()
                return (out_data, pyaudio.paAbort)

        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=SAMPLE_RATE,
                                  output=True,
                                  frames_per_buffer=L,
                                  stream_callback=callback)

    def play_audio(self):
        """Callback function for Play/Stop button."""

        if self.play_mode:
            self.play_button.config(text='Play')
            self.stream.stop_stream()
            self.stream.close()
            self.play_mode = False
        else:
            self.init_stream()
            self.play_mode = True
            self.play_button.config(text='Stop')
            self.stream.start_stream()


if __name__ == "__main__":
    app = App()
    app.mainloop()
