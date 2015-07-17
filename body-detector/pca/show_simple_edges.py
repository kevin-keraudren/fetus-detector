#!/usr/bin/env python
"""
http://www.vtk.org/Wiki/VTK/Examples/Python/Widgets/EmbedPyQt
http://stackoverflow.com/questions/3900253/vtk-how-can-i-add-a-scrollbar-to-my-project
http://stackoverflow.com/questions/18031555/how-to-connect-slider-to-a-function-in-pyqt4
"""

import sys
import vtk
from PyQt4 import QtCore, QtGui
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from vtk.util.colors import red, peacock

import numpy as np
from sklearn.decomposition import PCA

labels = [ ["brain"], # 0
           ["heart"], # 1
           ["left_lung","right_lung"], # 2
           ["liver","left_kidney","right_kidney"], # 3
           ["left_eye","right_eye"], # 4
           ["neck"] ] # 5

edges = [[0,4], # brain-eyes
         [0,5], # brain-neck
         [5,1], # neck-heart
         [1,2], # heart-lungs
         [1,3]  # heart-abdomen
         ]

colors = [ [ 255, 0, 0 ],        
           [ 0, 255, 0 ],       
           [ 0, 0, 255 ],        
           [ 255, 255, 0 ],     
           [ 0, 255, 255 ],       
           [ 255, 0, 255 ] ]

class MainWindow(QtGui.QMainWindow):

    def update_model( self, render=True ):
        model = self.pca.mean_.copy()
        for i in range(self.pca.components_.shape[0]):
            model += self.weights[i] * self.pca.components_[i]
        print model
        for i,s in enumerate(self.sources):
            s.SetCenter( model[3*i],
                         model[3*i+1],
                         model[3*i+2] )
            self.poly.GetPoints().SetPoint(i,
                                 model[3*i],
                                  model[3*i+1],
                                  model[3*i+2])

        # get a new tube filter
        self.tubes = vtk.vtkTubeFilter()
        self.tubes.SetInput(self.poly)
        self.tubes.SetRadius(0.1)
        self.tubes.SetNumberOfSides(6)
        self.mappers.append( vtk.vtkPolyDataMapper() )
        self.mappers[0].SetInputConnection(self.tubes.GetOutputPort())
        self.actors[0].SetMapper(self.mappers[0])
        
        if render:
            self.vtkWidget.Render()
        return

    def callback0(self, value):
        self.weights[0] =  float(value)/100.0
        self.update_model()
        return
    def callback1(self, value):
        self.weights[1] =  float(value)/100.0
        self.update_model()
        return
    def callback2(self, value):
        self.weights[2] =  float(value)/100.0
        self.update_model()
        return
    def callback3(self, value):
        self.weights[3] =  float(value)/100.0
        self.update_model()
        return
    def callback4(self, value):
        self.weights[4] =  float(value)/100.0
        self.update_model()
        return
    def callback5(self, value):
        self.weights[5] =  float(value)/100.0
        self.update_model()
        return
    def __init__(self, parent = None):
        self.sources = []

        data = np.load(sys.argv[1])

        # remove radius information
        print data.shape
        data = np.delete( data, np.arange(3, data.shape[0], 4), axis=1)
        print data.shape

        print data, data.shape
        self.pca = PCA()
        self.pca.fit( data )
        print self.pca.explained_variance_ratio_

        self.weights = np.zeros(self.pca.components_.shape[0], dtype='float')
        
        QtGui.QMainWindow.__init__(self, parent)
 
        self.frame = QtGui.QFrame()
 
        self.vl = QtGui.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)
 
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.slider0 = QtGui.QSlider(self.frame)
        self.slider0.setOrientation(QtCore.Qt.Horizontal)
        self.slider0.setObjectName("slider0")
        self.slider0.setRange(-100,100)
        self.vl.addWidget(self.slider0)
        QtCore.QObject.connect( self.slider0,
                                QtCore.SIGNAL("valueChanged(int)"), 
                                self.callback0 )

        self.slider1 = QtGui.QSlider(self.frame)
        self.slider1.setOrientation(QtCore.Qt.Horizontal)
        self.slider1.setObjectName("slider1")
        self.slider1.setRange(-100,100)
        self.vl.addWidget(self.slider1)
        QtCore.QObject.connect( self.slider1,
                                QtCore.SIGNAL("valueChanged(int)"), 
                                self.callback1 )

        self.slider2 = QtGui.QSlider(self.frame)
        self.slider2.setOrientation(QtCore.Qt.Horizontal)
        self.slider2.setObjectName("slider2")
        self.slider2.setRange(-100,100)
        self.vl.addWidget(self.slider2)
        QtCore.QObject.connect( self.slider2,
                                QtCore.SIGNAL("valueChanged(int)"), 
                                self.callback2 )

        self.slider3 = QtGui.QSlider(self.frame)
        self.slider3.setOrientation(QtCore.Qt.Horizontal)
        self.slider3.setObjectName("slider3")
        self.slider3.setRange(-100,100)
        self.vl.addWidget(self.slider3)
        QtCore.QObject.connect( self.slider3,
                                QtCore.SIGNAL("valueChanged(int)"), 
                                self.callback3 )

        self.slider4 = QtGui.QSlider(self.frame)
        self.slider4.setOrientation(QtCore.Qt.Horizontal)
        self.slider4.setObjectName("slider4")
        self.slider4.setRange(-100,100)
        self.vl.addWidget(self.slider4)
        QtCore.QObject.connect( self.slider4,
                                QtCore.SIGNAL("valueChanged(int)"), 
                                self.callback4 )

        self.slider5 = QtGui.QSlider(self.frame)
        self.slider5.setOrientation(QtCore.Qt.Horizontal)
        self.slider5.setObjectName("slider5")
        self.slider5.setRange(-100,100)
        self.vl.addWidget(self.slider5)
        QtCore.QObject.connect( self.slider5,
                                QtCore.SIGNAL("valueChanged(int)"), 
                                self.callback5 )

        self.mappers = []
        self.actors = []
        
        # Create source
        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName( "Colors")

        for l, c in zip(labels,colors):
            id = self.points.InsertNextPoint( 0,0,0 )
            print "ID", id

        for i in range(0,len(edges)):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0,edges[i][0])
            line.GetPointIds().SetId(1,edges[i][1])
            self.lines.InsertNextCell(line)
            vtk_colors.InsertNextTuple3( 0,153,76 )
        
        self.poly = vtk.vtkPolyData()
        self.poly.SetPoints(self.points)
        self.poly.SetLines(self.lines)
        self.poly.GetCellData().SetScalars(vtk_colors)
        self.poly.Update()
    
        # cleaner = vtk.vtkCleanPolyData()
        # cleaner.SetInput(self.poly)

        self.tubes = vtk.vtkTubeFilter()
        self.tubes.SetInput(self.poly)
        self.tubes.SetRadius(0.1)
        self.tubes.SetNumberOfSides(6)
        self.mappers.append( vtk.vtkPolyDataMapper() )
        self.mappers[-1].SetInputConnection(self.tubes.GetOutputPort())
        self.actors.append( vtk.vtkActor() )
        self.actors[-1].SetMapper(self.mappers[-1])
        self.actors[-1].GetProperty().SetSpecularColor(1, 1, 1)
        self.actors[-1].GetProperty().SetSpecular(0.3)
        self.actors[-1].GetProperty().SetSpecularPower(20)
        self.actors[-1].GetProperty().SetAmbient(0.2)
        self.actors[-1].GetProperty().SetDiffuse(0.8)
        self.ren.AddActor(self.actors[-1])
    
        for l, c in zip(labels,colors):
            self.sources.append( vtk.vtkSphereSource() )
            self.sources[-1].SetCenter(0, 0, 0)
            self.sources[-1].SetRadius(0.5)

            self.sources[-1].SetThetaResolution(30)
            self.sources[-1].SetPhiResolution(30)
 
            # Create a mapper
            self.mappers.append( vtk.vtkPolyDataMapper() )
            self.mappers[-1].SetInputConnection(self.sources[-1].GetOutputPort())
 
            # Create an actor
            self.actors.append( vtk.vtkActor() )
            self.actors[-1].SetMapper(self.mappers[-1])

            self.actors[-1].GetProperty().SetColor(c) # (R,G,B)
            #self.actors[-1].GetProperty().SetOpacity(0.5)
            
            self.ren.AddActor(self.actors[-1])

        self.update_model(render=False)
 
        self.ren.ResetCamera()
 
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
 
        self.show()
        self.iren.Initialize()
 
 
if __name__ == "__main__":
 
    app = QtGui.QApplication(sys.argv)
 
    window = MainWindow()
 
    sys.exit(app.exec_())
