import sys        
import logging
import numpy as np
import time

from PySide2 import QtCore      # core Qt functionality
from PySide2 import QtWidgets
from PySide2 import QtGui # extends QtCore with GUI functionality
from PySide2.QtOpenGL import QGLWidget as QOpenGLWidget    # provides QGLWidget, a special OpenGL QWidget
from PySide2.QtWidgets import QWidget,QStyleOptionSlider,QSizePolicy,QSlider,QStyle,QApplication,QCheckBox,QLabel,QHBoxLayout,QTextEdit,QPushButton
from PySide2.QtGui import QPainter,QPalette,QBrush,QPaintEvent,QMouseEvent,QKeySequence
from PySide2.QtCore import Qt,QSize,QRect
from PySide2 import QtOpenGL
from matplotlib import pyplot as plt
from matplotlib import cm
import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality
from OpenGL.arrays import vbo



log = logging.getLogger('root')
class RangeSlider(QSlider):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setValue(0.0)
        self.first_position = 0
        self.second_position = 100

        self.opt = QStyleOptionSlider()
        self.opt.minimum = 0.0
        self.opt.maximum = 100.0

        self.setTickPosition(QSlider.TicksBelow)
        self.setTickInterval(1)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed, QSizePolicy.Slider)
        )

    def setRangeLimit(self, minimum: int, maximum: int):
        self.opt.minimum = minimum
        self.opt.maximum = maximum

    def setRange(self, start: int, end: int):
        self.first_position = start
        self.second_position = end

    def getRange(self):
        return (self.first_position, self.second_position)

    def setTickPosition(self, position: QSlider.TickPosition):
        self.opt.tickPosition = position

    def setTickInterval(self, ti: int):
        self.opt.tickInterval = ti

    def paintEvent(self, event: QPaintEvent):

        painter = QPainter(self)

        # Draw rule
        self.opt.initFrom(self)
        self.opt.rect = self.rect()
        self.opt.sliderPosition = 0
        self.opt.subControls = QStyle.SC_SliderGroove | QStyle.SC_SliderTickmarks

        #   Draw GROOVE
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

        #  Draw INTERVAL

        color = self.palette().color(QPalette.Highlight)
        color.setAlpha(160)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)

        self.opt.sliderPosition = self.first_position
        x_left_handle = (
            self.style()
            .subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
            .right()
        )

        self.opt.sliderPosition = self.second_position
        x_right_handle = (
            self.style()
            .subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
            .left()
        )

        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, self.opt, QStyle.SC_SliderGroove
        )

        selection = QRect(
            x_left_handle,
            groove_rect.y(),
            x_right_handle - x_left_handle,
            groove_rect.height(),
        ).adjusted(-1, 1, 1, -1)

        painter.drawRect(selection)

        # Draw first handle

        self.opt.subControls = QStyle.SC_SliderHandle
        self.opt.sliderPosition = self.first_position
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

        # Draw second handle
        self.opt.sliderPosition = self.second_position
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

    def mousePressEvent(self, event: QMouseEvent):
        self.opt.sliderPosition = self.first_position
        pos = QtCore.QPoint()
        position = event.pos()
        pos.setX(int(position.x()))
        pos.setY(int(position.y()))
        self._first_sc = self.style().hitTestComplexControl(
            QStyle.CC_Slider, self.opt, pos, self
        )

        self.opt.sliderPosition = self.second_position
        self._second_sc = self.style().hitTestComplexControl(
            QStyle.CC_Slider, self.opt, pos, self
        )        

    def mouseMoveEvent(self, event: QMouseEvent):

        distance = self.opt.maximum - self.opt.minimum
        position = event.pos()
        pos = self.style().sliderValueFromPosition(
            0, distance, position.x(), self.rect().width()
        )

        self.trigger_value_change()
        if self._first_sc == QStyle.SC_SliderHandle:
            if pos <= self.second_position:
                self.first_position = pos
                self.update()
                return

        if self._second_sc == QStyle.SC_SliderHandle:
            if pos >= self.first_position:
                self.second_position = pos
                self.update()

    def trigger_value_change(self):
        v = self.value()
        if v == 0.0:
            self.setValue(1.0)
        else:
            self.setValue(0.0)
            
    def sizeHint(self):
        """ override """
        SliderLength = 84
        TickSpace = 5

        w = SliderLength
        h = self.style().pixelMetric(QStyle.PM_SliderThickness, self.opt, self)

        if (
            self.opt.tickPosition & QSlider.TicksAbove
            or self.opt.tickPosition & QSlider.TicksBelow
        ):
            h += TickSpace

        return (
            self.style()
            .sizeFromContents(QStyle.CT_Slider, self.opt, QSize(w, h), self)
#            .expandedTo(QApplication.globalStrut())
        )

    
class GLWidget(QOpenGLWidget):
    def __init__(self,vertices, indices, pixel_mask, init_data = False, parent=None):
        QOpenGLWidget.__init__(self, parent)
        
        if isinstance(init_data,bool) and (not isinstance(pixel_mask,bool)) :
            self._data = np.full(pixel_mask.shape,0.0)
            self._data[pixel_mask]=1.0    
        elif isinstance(init_data,np.ndarray):
            self._data = np.zeros(pixel_mask.shape)
            self._data[pixel_mask] = init_data.flatten()
        elif isinstance(vertices,bool) or isinstance(indices,bool) or isinstance(pixel_mask,bool):
            vertices,indices,init_data,pixel_mask = self.initGeometry()
            self._data = init_data
            
        self.mouse_is_dragged = False
        self.background_color = (0.2,0.2,0.2,0.0)
        self.low_clip_color = (0.4,0.4,0.4,0.0)
        self.high_clip_color = (0.0,1,0.0,0.0)
        self.view_angle = 45.0
        self._use_log_color_scale = True
        self._use_absolute_color_range = False
        self._absolute_color_limits = [0,100]
        self.current_color_limits = self._absolute_color_limits
        self._colormap= cm.get_cmap('plasma')
        self.parent = parent
        self.pixel_mask = pixel_mask
        #vertices,indices,colors = self.initGeometry()
        self._vertices = vertices
        self.create_view_port()
       
        self._colormap_range = (0.,1.)
        self._colors = self.data_to_color()
        self._indices = indices
        #print('shapes v {} i {} c {}'.format(self._vertices.shape,self._indices.shape, self._colors.shape))
        self.create_view_port()
        self.create_vbo()
        self.rendered_frames = 0
        self.fps_last_checked = time.time() 
        
        #gl.glClearColor(QtGui.QColor(0, 0, 255,255))    # initialize the screen to blue
        
 
    def create_view_port(self):
        verts = self.vertices
        center = np.mean(verts,axis = 0)
        #print('center = {}'.format(center))
        new_verts = verts - center
        max_dimension = np.max(np.linalg.norm(new_verts, axis = -1))
        self.max_dimension = max_dimension
        #print('max dimension = {}'.format(max_dimension))
        min_distance = (2*max_dimension)/np.tan(self.view_angle/360*2*np.pi)
        
        #print('min distance = {}'.format(min_distance))
        margin=-1
        max_distance = min_distance + (2+margin)*max_dimension
        #print('max distance = {}'.format(max_distance))
        object_z_shift = min_distance + (2+margin)/2*max_dimension
        object_shift = [0.0, 0.0, -object_z_shift]
        #print('object_distance distance = {}'.format(object_z_shift))
        self.viewport = [min_distance/100,max_distance*100,object_shift,-center]
        #print(self.viewport)                
        
        
                  
    def set_frame(self, vertices,data,indices):
        """Load 2D data as a Nx2 Numpy array.
        """
        self.vertices = vertices.astype(np.float32)
        self.data = data.astype(np.float32)
        self.indices = indices.astype(np.uint32)
        #vertices,indices,colors = self.initGeometry()
        #self.vertices = vertices
        #self.indices = indices
        
    def create_vc_vbo(self):
        vc = np.concatenate((self._vertices,self._colors),axis = 1).astype(np.float32)
        self.vc_VBO = vbo.VBO(vc)

    def create_index_vbo(self):
        self.index_VBO = vbo.VBO(self._indices,target='GL_ELEMENT_ARRAY_BUFFER')

    def data_to_color(self):
        if self._use_absolute_color_range:
            #print('use absolute color definition')
            colors = self.data_to_color_absolute()
        else:
            #print('use relative color definition')
            colors = self.data_to_color_relative()
        return colors
    def data_to_color_absolute(self):
        data = self._data
        length = self._absolute_color_limits[1] - self._absolute_color_limits[0]
        c_range = self._colormap_range
        view_min = self._absolute_color_limits[0] + c_range[0]*length
        view_max = self._absolute_color_limits[0] + c_range[1]*length
        view_length = view_max - view_min
        if view_length == 0:
            view_min = view_min*(1-1/2)
            view_max = view_max*(1+1/2)
            view_length = 1
        if self._use_log_color_scale:
            if view_max <=0:
                log.warning('All Values 0 or negative detected! They will be displayed as 0.')
                colors = np.full((len(data),4),0.0)
            else:
                if view_min<0:
                    length = self._absolute_color_limits[1]
                    view_min = c_range[0]*length
                    view_max = c_range[1]*length
                    view_length = view_max - view_min
                    log.warning('negative Values detected! They will be displayed as 0.')
                    
            view_data = 1+(data - view_min)/view_length*(np.exp(1)-1)
            low_data_mask = (view_data<=1)
            high_data_mask = (view_data>np.exp(1))
            view_data[low_data_mask] = 1.0
            view_data[high_data_mask] = np.exp(1)
            view_data = np.log(view_data)
            #print("view_data max = {} min = {}".format(view_data.max(),view_data.min()))
            colors = self._colormap(view_data)
            colors[low_data_mask,:] = self.low_clip_color
            colors[high_data_mask,:] = self.high_clip_color
        else:
            colors = self._colormap((data-view_min)/view_length)
        self.current_color_limits = [view_min,view_max]
        return colors.reshape(-1,4)

    def data_to_color_relative(self):
        data = self._data
        d_max = data.max()
        d_min = data.min()
        c_range = self._colormap_range
        length = d_max-d_min
        view_min = d_min + c_range[0]*length
        view_max = d_min + c_range[1]*length
        view_length = view_max - view_min
        if view_length == 0:
            view_min = view_min*(1-1/2)
            view_max = view_max*(1+1/2)
            view_length = 1
        if self._use_log_color_scale:
            if view_max <=0:
                log.warning('All Values 0 or negative detected! They will be displayed as 0.')
                colors = np.full((len(data),4),0.0)
            else:
                if view_min<0:
                    length = d_max
                    view_min = c_range[0]*length
                    view_max = c_range[1]*length
                    view_length = view_max - view_min
                    log.warning('negative Values detected! They will be displayed as 0.')

            view_data = 1+(data - view_min)/view_length*(np.exp(1)-1)
            low_data_mask = (view_data<=1)
            high_data_mask = (view_data>np.exp(1))
            view_data[low_data_mask] = 1.0
            view_data[high_data_mask] = np.exp(1)
            view_data = np.log(view_data)
            #print("view_data max = {} min = {}".format(view_data.max(),view_data.min()))
            colors = self._colormap(view_data)
            colors[low_data_mask,:] = self.low_clip_color
            colors[high_data_mask,:] = self.high_clip_color
        else:
            colors = self._colormap((data-view_min)/view_length)
        self.current_color_limits = [view_min,view_max]
        self._absolute_color_limits = [view_min,view_max]
        #print('colors shape ={}'.format(colors.shape))
        return colors.reshape(-1,4)
        
    def initializeGL(self):
        gl.glClearColor(*self.background_color)
        gl.glEnable(gl.GL_DEPTH_TEST)                  # enable depth testing
        gl.glShadeModel(gl.GL_FLAT)

        #self.initGeometry()
        #self.create_vbo()

        self.rotX = 180.0
        self.rotY = 0.0
        self.rotZ = 0.0
         
    def resizeGL(self, width, height):
        viewport = self.viewport
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        #gl.glOrtho(-500, 500, 500, -500, -2000, 2000)
        aspect = width / float(height)

        GLU.gluPerspective(self.view_angle, aspect, viewport[0], viewport[1])
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        self.print_fps()
        #start = time.time()
        viewport =self.viewport
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glPushMatrix()    # push the current matrix to the current stack

        gl.glTranslate(*viewport[2])    # third, translate cube to specified depth
        #gl.glScale(20.0, 20.0, 20.0)       # second, scale cube
        gl.glRotate(self.rotX, 1.0, 0.0, 0.0)
        gl.glRotate(self.rotY, 0.0, 1.0, 0.0)
        gl.glRotate(self.rotZ, 0.0, 0.0, 1.0)
        gl.glTranslate(*viewport[3])   # first, translate cube center to origin

        self.vc_VBO.bind()
        #self.vbo.bind()
#        self.colorVBO.bind()
        self.index_VBO.bind()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        
        gl.glColorPointer(4, gl.GL_FLOAT, 28, self.vc_VBO+12)
        gl.glVertexPointer(3, gl.GL_FLOAT, 28, self.vc_VBO)
        
        #gl.glDrawElements(gl.GL_TRIANGLES, len(self._indices), gl.GL_UNSIGNED_INT, None)
        gl.glDrawElements(gl.GL_QUADS, len(self._indices), gl.GL_UNSIGNED_INT, None)
        #gl.glDrawElements(gl.GL_POINTS, len(self._indices), gl.GL_UNSIGNED_INT, None)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glPopMatrix()    # restore the previous modelview matrix
        #stop = time.time()
        #print('fps according to opengl = {}'.format(1/(stop-start)))
    def print_fps(self):
        rendered_frames = self.rendered_frames
        rendered_frames += 1
        current_time = time.time()
        elapsed_time = current_time-self.fps_last_checked
        if elapsed_time >=1:
            print('fps = {}'.format(rendered_frames))
            rendered_frames = 0
            self.fps_last_checked = current_time
        self.rendered_frames = rendered_frames
        
    def create_vbo(self):
        self.create_vc_vbo()
        self.create_index_vbo()

    def initGeometry(self):
        vertices = np.array(
                [[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [0.0, 1.0, 1.0]]).astype(np.float32)

        colors = np.array(
                [[0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [1.0, 0.0, 1.0, 0.0],
                 [1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 0.0 ]]).astype(np.float32)
        data = np.arange(8,dtype = np.float32)
        mask = np.ones(8,dtype = bool)
        indices = np.array(
                [0, 1, 2, 3,
                 3, 2, 6, 7,
                 1, 0, 4, 5,
                 2, 1, 5, 6,
                 0, 3, 7, 4,
                 7, 6, 5, 4 ]).astype(np.int32)
        
        return vertices,indices,data,mask
        
    def setRotX(self, val):
        self.rotX = 360 * val + 180

    def setRotY(self, val):
        self.rotY = 360 * val

    def setRotZ(self, val):
        self.rotZ = 360 * val

    def addRotX(self, val):
        self.rotX += 360 * val
    def addRotY(self, val):
        self.rotY += 360 * val

    def addRotZ(self, val):
        self.rotZ += 360 * val

    
    def wheelEvent(self,event):
        min_distance = self.viewport[0]
        degree_change = event.angleDelta().y()/(8*360)*np.pi*2
        self.viewport[2][2] += degree_change*min_distance*10

    def keyPressEvent(self,event):
        step = self.max_dimension/10        
        if event.matches(Qt.Key_W):
            self.viewport[2][0] -= step
        elif event.matches(Qt.Key_S):
            self.viewport[2][0] -= step
        elif event.matches(Qt.Key_A):
            self.viewport[2][1] += step
        elif event.matches(Qt.Key_D):
            self.viewport[2][1] -= step
        
    def mouseMoveEvent(self, event):
        if self.mouse_is_dragged:
            x_start,y_start = self.mousepress_start_rotation_state[:2]
            old_pos = self.current_pos
            new_pos = event.pos()

            delta = [old_pos.x()-new_pos.x(), old_pos.y() - new_pos.y()]
            dampening = 10000
            x_rot_val = x_start + (-delta[1]/dampening)%1
            y_rot_val = (delta[0]/dampening)%1
        self.setRotX(x_rot_val)
        self.setRotY(y_rot_val)
            
    def mousePressEvent(self,event):
        self.current_pos=event.pos()
        self.mousepess_start_rotation_state = [self.rotX,self.rotY,self.rotZ]
        self.mouse_is_dragged=True
    def mouseReleaseEvent(self,event):
        self.mouse_is_dragged=False

    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self,vertices):
        self._vertices = vertices
        self.create_vc_vbo()        
        
    @property
    def indices(self):
        return self._indices
    @indices.setter
    def indices(self,indices):
        self._indices = indices
        self.create_vbo()
        
    @property
    def use_absolute_color_range(self):
        return self._use_absolute_color_range
    @use_absolute_color_range.setter
    def use_absolute_color_range(self,use_absolute_color_range):
        self._use_absolute_color_range = use_absolute_color_range
        #print('abs colors checkbox is changed')
        self.colors = self.data_to_color()

    @property
    def absolute_color_limits(self):
        return self._absolute_color_limits
    @absolute_color_limits.setter
    def absolute_color_limits(self,values):
        self._absolute_color_limits = values
        #print('abs colors range is changed')
        if self._use_absolute_color_range:
            #print('abs colors range is applied changed')
            self.colors = self.data_to_color()
        
    @property
    def data(self):
        return self.data
    @data.setter
    def data(self,data):
        self._data[self.pixel_mask] = data.flatten()
        self.colors = self.data_to_color()
        self.create_vc_vbo()

    @property
    def colors(self):
        return self.colors
    @colors.setter
    def colors(self,colors):
        self._colors = colors
        self.create_vc_vbo()

    @property
    def use_log_color_scale(self):
        return self._use_log_color_scale
    
    @use_log_color_scale.setter
    def use_log_color_scale(self,use_log_color_scale):
        self._use_log_color_scale = use_log_color_scale
        self.colors = self.data_to_color()

    @property
    def colormap(self):
        return self._colormap
    @colormap.setter
    def colormap(self,colormap_name):
        self._colormap = cm.get_cmap(colormap_name)
        self.colors = self.data_to_color()

    @property
    def colormap_range(self):
        return self._colormap_range
    @colormap_range.setter
    def colormap_range(self,c_range):
        color_range_changed = ~(np.isclose(c_range[0], self._colormap_range[0]) & np.isclose(c_range[1],self._colormap_range[1]))
        if color_range_changed:
            self._colormap_range = c_range
            self.colors = self.data_to_color()
            

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, pixel_vertices = False ,pixel_indices = False,sensitive_pixel_mask = 0):
        n_points=int(1e4)
        n_pics = int(1e3)
        if isinstance(pixel_vertices,bool):
            scale=0.2
            vertices = scale*np.random.randn(n_points,3).astype(np.float32)
            vertices[:,2]=0
            self.pixel_vertices = vertices
        else:
            self.pixel_vertices = pixel_vertices
            
        if isinstance(pixel_indices,bool):
            indices = np.arange(n_points,dtype=np.uint32)
            self.pixel_indices = indices
        else:
            self.pixel_indices = pixel_indices
        self.pixel_mask = sensitive_pixel_mask
        
        QtWidgets.QMainWindow.__init__(self)    # call the init for the parent class
        
        self.resize(300, 300)
        self.setWindowTitle('Hello OpenGL App')

        self._datasets = []
        self._meta_data=False
            
        self.glWidget = GLWidget(self.pixel_vertices, self.pixel_indices, self.pixel_mask)
        self.initGUI()
        
        timer = QtCore.QTimer(self)
        timer.setInterval(20)   # period, in milliseconds
        timer.timeout.connect(self.glWidget.update)
        timer.start()

    @property
    def meta_data(self):
        return meta_data
    @meta_data.setter
    def meta_data(self,_dict):
        self._meta_data = _dict
        
    @property
    def datasets(self):
        return self._datasets
    @datasets.setter
    def datasets(self,datasets):
        self._datasets = datasets
        self.pics_slider.setMaximum(max(0,len(datasets)-1))
        self.change_pic(0)
        
    def generate_data(self,n_points,n_pics):
        datasets = [np.random.rand(n_points).astype(np.float32) for i in range(n_pics)]
        return datasets
        
    def initGUI(self):
        central_widget = QtWidgets.QWidget()
        gui_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(gui_layout)
        self.setCentralWidget(central_widget)
        
        gui_layout.addWidget(self.glWidget)

        sliderX = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderX.valueChanged.connect(lambda val: self.glWidget.setRotX(float(val)/99))

        sliderY = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderY.valueChanged.connect(lambda val: self.glWidget.setRotY(float(val)/99))

        sliderZ = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderZ.valueChanged.connect(lambda val: self.glWidget.setRotZ(float(val)/99))

        slider_pics = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_pics.valueChanged.connect(self.change_pic)
        slider_pics.setMinimum(0)
        slider_pics.setMaximum(max(0,len(self.datasets)-1))
        slider_pics.setSingleStep(1)
        self.pics_slider = slider_pics

        self.range_slider = RangeSlider()
        self.range_slider.valueChanged.connect(self.change_colormap_region)

        log_scale_check_box = QCheckBox('Log color scale')
        log_scale_check_box.stateChanged.connect(self.change_log_scale)
        log_scale_check_box.setChecked(True)

        title_label = QLabel()
        title_label.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        title_label.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Maximum)
        self.title_label = title_label
        

        absolute_color_widget = QHBoxLayout()
        abs_col_check_box = QCheckBox('Absolute color range')
        abs_col_check_box.stateChanged.connect(self.change_use_abs_color_range)
        abs_col_min_text_edit = QTextEdit('')
        tmp = abs_col_min_text_edit
        textHeight = tmp.document().documentLayout().documentSize().height()
        tmp.setFixedHeight(textHeight + tmp.height() - tmp.viewport().height())
        self.abs_col_min_text_edit = abs_col_min_text_edit
        abs_col_max_text_edit = QTextEdit('')
        tmp = abs_col_max_text_edit
        abs_col_max_text_edit.setFixedHeight(textHeight + tmp.height() - tmp.viewport().height())
        self.abs_col_max_text_edit = abs_col_max_text_edit
        abs_color_apply_button = QPushButton('Apply')
        abs_color_apply_button.clicked.connect(self.change_abs_color_range)
        current_col_min_label = QLabel('Color Range Min:  ')
        self.current_col_min_label = current_col_min_label
        current_col_max_label = QLabel('Color Range Max:  ')
        self.current_col_max_label = current_col_max_label

        current_col_max_label.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)
        current_col_min_label.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)
        abs_col_max_text_edit.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)
        abs_col_min_text_edit.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)
        abs_col_check_box.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)
        abs_color_apply_button.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)
        
        absolute_color_widget.addWidget(abs_col_check_box)
        absolute_color_widget.addWidget(abs_col_min_text_edit)
        absolute_color_widget.addWidget(abs_col_max_text_edit)
        absolute_color_widget.addWidget(abs_color_apply_button)
        absolute_color_widget.addWidget(current_col_min_label)
        absolute_color_widget.addWidget(current_col_max_label)
                
        gui_layout.addWidget(title_label)
        gui_layout.addWidget(slider_pics)
        gui_layout.addLayout(absolute_color_widget)
        gui_layout.addWidget(self.range_slider)
        gui_layout.addWidget(log_scale_check_box)
        gui_layout.addWidget(sliderX)
        gui_layout.addWidget(sliderY)
        gui_layout.addWidget(sliderZ)

    def update_abs_color_range_text(self):
        limits = self.glWidget.absolute_color_limits
        self.abs_col_min_text_edit.setPlainText(str(limits[0]))
        self.abs_col_max_text_edit.setPlainText(str(limits[1]))                                           

    def update_current_color_range_label(self):
        limits = self.glWidget.current_color_limits
        self.current_col_min_label.setText('Color Range Min: {}'.format(str(limits[0])))
        self.current_col_max_label.setText('Color Range Max: {}'.format(str(limits[1])))
        
    def change_abs_color_range(self):
        _min = 0
        _max = 100
        try:
            _min = float(self.abs_col_min_text_edit.toPlainText())
            _max = float(self.abs_col_max_text_edit.toPlainText())
        except ValueError as e:
            log.error(e)
            log.warning('Using default absolute color min = {} ,max = {}'.format(_min,_max))
        if self.glWidget.use_log_color_scale:
            if _min < 0:
                log.warning(' min of absolute color range < 0. Is not applicable for logscale. Setting min = 0.')
                _min = 0
        self.glWidget.absolute_color_limits = [_min,_max]
        self.update_current_color_range_label()
                
                
    def change_use_abs_color_range(self,is_checked):        
        self.glWidget.use_absolute_color_range = bool(is_checked)
        self.update_abs_color_range_text()
        self.update_current_color_range_label()

    def change_colormap_region(self,*args):        
        c_range=self.range_slider.getRange()
        c_range = (float(c_range[0])/100,float(c_range[1])/100)
        self.glWidget.colormap_range=c_range
        self.update_current_color_range_label()
        
                
    def change_pic(self,val):
        datasets = self.datasets
        pic_id=int(val)
        self.change_title_label(pic_id)
        dataset = datasets[pic_id]
        self.glWidget.data = dataset
        self.change_colormap_region()
        if not self.glWidget.use_absolute_color_range:
            self.update_current_color_range_label()
            self.update_abs_color_range_text()
        
    def change_title_label(self, pic_id):
        if isinstance(self._meta_data,dict):
            text = ''
            tab = '    '
            for key,vals in self._meta_data.items():
                current_val = vals[pic_id]
                text += key + ': {}'.format(current_val) + tab
            text = text[:-len(tab)]
            self.title_label.setText(text)
            

    def keyPressEvent(self,event):
        gl_window = self.glWidget
        step = gl_window.max_dimension/50
        if event.text() == 'a':
            gl_window.viewport[2][0] -= step
        elif event.text() == 'd':
            gl_window.viewport[2][0] += step
        elif event.text() == 'w':
            gl_window.viewport[2][1] += step
        elif event.text() == 's':
            gl_window.viewport[2][1] -= step
        elif event.text() == 'h':
            gl_window.viewport[2][0] = 0.0
            gl_window.viewport[2][1] = 0.0


    def change_log_scale(self,is_checked):
        self.glWidget.use_log_color_scale = bool(is_checked)
        
class FramePresenter():
    def __init__(self,pixel_vertices=False,pixel_indices=False,sensitive_pixel_mask=False):
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = QtWidgets.QApplication.instance()
        self.win = MainWindow(pixel_vertices = pixel_vertices,pixel_indices = pixel_indices,sensitive_pixel_mask = sensitive_pixel_mask)
        
    def show(self,datasets, meta_data = False):
        self.win.meta_data = meta_data
        self.win.datasets = datasets
        self.win
        self.win.show() 
        self.app.exec_()
        #print('fi')
