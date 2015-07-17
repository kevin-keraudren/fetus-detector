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

import numpy as np
from sklearn.decomposition import PCA

# class myPCA:
#     def __init__(self, n_components=None ):
#         self.n_components = n_components

#     def fit(X):
#         """
#         https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/pca.py
#         """
#         X = np.array( X, dtype='float' )
#         n_samples, n_features = X.shape
#         self.mean_ = X.mean(axis=0)
#         X -= self.mean_
#         U, S, V = np.linalg.svd(X, full_matrices=False)
#         explained_variance = (S ** 2) / n_samples
#         explained_variance_ratio = (explained_variance /
#                                     explained_variance.sum())

#         n_components = self.n_components
#         if n_components is None:
#             n_components = n_features
        
#         self.components_ = V
#         self.singular_values_ = S
        
#         return mean, components, singular_values, explained_variance_ratio

labels = [ ["brain"],
           ["heart"],
           ["left_lung","right_lung"],
           ["liver","left_kidney","right_kidney"],
           ["left_eye","right_eye"],
           ["neck"] ]

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
            s.SetCenter( model[4*i],
                         model[4*i+1],
                         model[4*i+2] )
            if i < 4:
                s.SetRadius(model[4*i+3])
            else:
                s.SetRadius(1.0)
        if render:
            self.vtkWidget.Render()
        return

    def callback0(self, value):
        self.weights[0] =  float(value)/100.0 * self.S[0]
        self.update_model()
        return
    def callback1(self, value):
        self.weights[1] =  float(value)/100.0 * self.S[1]
        self.update_model()
        return
    def callback2(self, value):
        self.weights[2] =  float(value)/100.0 * self.S[2]
        self.update_model()
        return
    def callback3(self, value):
        self.weights[3] =  float(value)/100.0 * self.S[3]
        self.update_model()
        return
    def callback4(self, value):
        self.weights[4] =  float(value)/100.0 * self.S[4]
        self.update_model()
        return
    def callback5(self, value):
        self.weights[5] =  float(value)/100.0 * self.S[5]
        self.update_model()
        return
    def __init__(self, parent = None):
        self.sources = []

        data = np.load(sys.argv[1])

        print data, data.shape
        self.pca = PCA()
        U,S,V = self.pca._fit( data )
        print self.pca.explained_variance_ratio_

        self.S = np.sqrt(S)#*3.0#/5.0

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
        for l, c in zip(labels,colors):
            # Create source
            self.sources.append( vtk.vtkSphereSource() )
            self.sources[-1].SetCenter(0, 0, 0)
            self.sources[-1].SetRadius(1.0)

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
