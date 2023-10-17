import os
import sys
import cv2
import time
import numpy as np
from PIL import Image
from util import util

import torch
from torchvision.utils import save_image
import torchvision

from options_vae.test_options import TestOptions
from data import create_dataloader
from models.models import create_model

from models.CA2SIS_model import CA2SISModel
from models.VAE_model import VAEModel

from data.base_dataset import BaseDataset, get_params, get_transform, normalize

from ui.ui_en import Ui_Form
from ui.mouse_event import GraphicsScene
from ui_util.config import Config

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

color_list = [QColor(0, 0, 0), QColor(204, 0, 0), QColor(76, 153, 0), QColor(204, 204, 0), QColor(51, 51, 255), QColor(204, 0, 204), QColor(0, 255, 255), QColor(51, 255, 255), QColor(102, 51, 0), QColor(255, 0, 0), QColor(102, 204, 0), QColor(255, 255, 0), QColor(0, 0, 153), QColor(0, 0, 204), QColor(255, 51, 153), QColor(0, 204, 204), QColor(0, 51, 0), QColor(255, 153, 51), QColor(0, 204, 0)]

class Ex(QWidget, Ui_Form):
    def __init__(self, vae_model, model, opt):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()
        self.model = model
        self.vae_model = vae_model
        self.opt = opt
        if len(opt.gpu_ids) > 0:
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        params = get_params(self.opt, (256,256))
        self.transform_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        self.transform_image = get_transform(self.opt, params)

        self.label_list = ['bkgrnd', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

        self.output_img = None

        self.mat_img = None

        self.mode = 0
        self.size = 6
        self.mask = None
        self.mask_m = None
        self.img = None

        self.mouse_clicked = False
        self.scene = GraphicsScene(self.mode, self.size)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ref_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.ref_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
 
        self.result_scene = QGraphicsScene()
        self.graphicsView_3.setScene(self.result_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.style_scene = QGraphicsScene()
        self.graphicsView_4.setScene(self.style_scene)
        self.graphicsView_4.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        
            if len(self.ref_scene.items())>0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            self.ref_scene.addPixmap(image)
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(image)


        mask_filename = fileName.replace("jpg", "png")  
        self.input_mask_rgb = cv2.imread(mask_filename)
        self.input_mask_rgb = cv2.resize(self.input_mask_rgb, (256,256), interpolation = cv2.INTER_NEAREST)


    def open_style(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            self.img_style = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        
            if len(self.style_scene.items())>0:
                self.style_scene.removeItem(self.style_scene.items()[-1])
            self.style_scene.addPixmap(image)


        mask_filename = fileName.replace("jpg", "png")  
        self.input_mask_rgb_style = cv2.imread(mask_filename)
        self.input_mask_rgb_style = cv2.resize(self.input_mask_rgb_style, (256,256), interpolation = cv2.INTER_NEAREST)



    def preprocess_mask(self, label_map):
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
        return input_label.scatter_(1, label_map, 1.0)
    

    def open_mask(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:   

            label = Image.open(fileName)
            params = get_params(self.opt, label.size)
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            label_tensor = transform_label(label) * 255.0
            label_tensor[label_tensor == 255] = self.opt.label_nc
            self.input_mask = self.preprocess_mask(label_tensor.unsqueeze(0).long())


            mat_img = cv2.imread(fileName)
            mat_img = cv2.resize(mat_img, (256,256), interpolation = cv2.INTER_NEAREST)
            
            self.mask = mat_img.copy()
            self.mask_m = mat_img       
            mat_img = mat_img.copy()
            image = QImage(mat_img, 256, 256, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return    

            for i in range(256):
                for j in range(256):
                    r, g, b, a = image.pixelColor(i, j).getRgb()
                    image.setPixel(i, j, color_list[r].rgb()) 
           
            pixmap = QPixmap()
            pixmap.convertFromImage(image)  
            self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.scene.reset()
            if len(self.scene.items())>0:
                self.scene.reset_items() 
            self.scene.addPixmap(self.image)

    def bg_mode(self):
        self.scene.mode = 0

    def skin_mode(self):
        self.scene.mode = 1

    def nose_mode(self):
        self.scene.mode = 2

    def eye_g_mode(self):
        self.scene.mode = 3

    def l_eye_mode(self):
        self.scene.mode = 4

    def r_eye_mode(self):
        self.scene.mode = 5

    def l_brow_mode(self):
        self.scene.mode = 6

    def r_brow_mode(self):
        self.scene.mode = 7

    def l_ear_mode(self):
        self.scene.mode = 8

    def r_ear_mode(self):
        self.scene.mode = 9

    def mouth_mode(self):
        self.scene.mode = 10

    def u_lip_mode(self):
        self.scene.mode = 11

    def l_lip_mode(self):
        self.scene.mode = 12

    def hair_mode(self):
        self.scene.mode = 13

    def hat_mode(self):
        self.scene.mode = 14

    def ear_r_mode(self):
        self.scene.mode = 15

    def neck_l_mode(self):
        self.scene.mode = 16

    def neck_mode(self):
        self.scene.mode = 17

    def cloth_mode(self):
        self.scene.mode = 18

    def increase(self):
        if self.scene.size < 15:
            self.scene.size += 1
    
    def decrease(self):
        if self.scene.size > 1:
            self.scene.size -= 1 

    def one_hot(self, targets, nclasses):
        targets_extend = targets.clone()        
        targets_extend.unsqueeze_(1)  # convert to Nx1xHxW
        one_hot = torch.FloatTensor(targets_extend.size(0), nclasses, targets_extend.size(2), targets_extend.size(3)).zero_()
        one_hot = one_hot.to(self.device)
        one_hot.scatter_(1, targets_extend, 1)
        return one_hot

    def convert_mask(self, mask):
        mask = Image.fromarray(np.uint8(mask))
        to_bw = torchvision.transforms.Grayscale()
        mask = to_bw(mask)
        mask = self.transform_mask(mask)*255.
        mask[mask == 255] = self.opt.label_nc 
        return mask    

    def generate_part(self):
        
        mask_m = self.mask_m.copy()

        mask_m = self.convert_mask(mask_m)

        mask_m_19 = self.preprocess_mask(mask_m.unsqueeze(0).long())

        p = self.scene.mode
        generated_m = self.vae_model.generate_parts(mask_m_19.to(self.device), [p])

        generated_m = torch.argmax(generated_m, dim=1)    
        result = np.asarray(generated_m.detach().cpu(), dtype=np.uint8).copy()
        result = np.squeeze(result, axis=0)
        
        self.mask_m = result

        #cv2.imwrite("asd.png", result)
        image = QImage(256, 256, QImage.Format_RGB888)
        

        for i in range(256):
            for j in range(256):
                r = result[j][i]

                image.setPixel(i, j, color_list[r].rgb()) 
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image)  
        image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        self.scene.reset()
        self.scene.mode = p
        if len(self.scene.items())>0:
            self.scene.reset_items() 
        self.scene.addPixmap(image)   

    def perturbate_part(self):
        
        mask_m = self.mask_m.copy()

        mask_m = self.convert_mask(mask_m)

        mask_m_19 = self.preprocess_mask(mask_m.unsqueeze(0).long())

        p = self.scene.mode
        generated_m = self.vae_model.generate_perturbations(mask_m_19.to(self.device), [p])

        generated_m = torch.argmax(generated_m, dim=1)    
        result = np.asarray(generated_m.detach().cpu(), dtype=np.uint8).copy()
        result = np.squeeze(result, axis=0)
        
        self.mask_m = result

        #cv2.imwrite("asd.png", result)
        image = QImage(256, 256, QImage.Format_RGB888)
        

        for i in range(256):
            for j in range(256):
                r = result[j][i]

                image.setPixel(i, j, color_list[r].rgb()) 
        
        pixmap = QPixmap()
        pixmap.convertFromImage(image)  
        image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        self.scene.reset()
        self.scene.mode = p
        if len(self.scene.items())>0:
            self.scene.reset_items() 
        self.scene.addPixmap(image)


    def edit(self):

        for i in range(19):
            self.mask_m = self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)

        mask = self.mask.copy()
        mask_m = self.mask_m.copy()

        mask_m = self.convert_mask(mask_m)

        mask = self.convert_mask(self.input_mask_rgb)
        
        img = self.transform_image(self.img)

        mask_19 = self.preprocess_mask(mask.unsqueeze(0).long())
        mask_m_19 = self.preprocess_mask(mask_m.unsqueeze(0).long())
        #print(mask.size(), mask_m.size(), img.size())

        start_t = time.time()
        generated = self.model.netG(img.unsqueeze(0).to(self.device),
                               mask_19.to(self.device), 
                               mask_m_19.to(self.device))

        # generated = model.inference(torch.FloatTensor([mask_m.numpy()]), torch.FloatTensor([mask.numpy()]), torch.FloatTensor([img.numpy()]))   
        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))
        #save_image((generated.data[0] + 1) / 2,'./results/1.jpg')

        # result = generated.permute(0, 2, 3, 1)
        # result = result.detach().cpu().numpy()
        # result = (result + 1) * 127.5

        result = util.tensor2im(generated[0].squeeze().detach().cpu())      
        result = np.asarray(result, dtype=np.uint8).copy()
        #qim = QImage(result.data, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)
        qim = QImage(result.data, result.shape[1], result.shape[0], QImage.Format_RGB888)

        #for i in range(256):
        #    for j in range(256):
        #       r, g, b, a = image.pixelColor(i, j).getRgb()
        #       image.setPixel(i, j, color_list[r].rgb()) 
        if len(self.result_scene.items())>0: 
            self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(QPixmap.fromImage(qim))

    def edit_style(self):
        for i in range(19):
            self.mask_m = self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)

        mask = self.mask.copy()
        mask_m = self.mask_m.copy()

        mask_m = self.convert_mask(mask_m)

        mask_style = self.convert_mask(self.input_mask_rgb_style)

        mask = self.convert_mask(self.input_mask_rgb)

        img = self.transform_image(self.img)
        img_style = self.transform_image(self.img_style)

        mask_19 = self.preprocess_mask(mask.unsqueeze(0).long())
        mask_m_19 = self.preprocess_mask(mask_m.unsqueeze(0).long())
        mask_style_19 = self.preprocess_mask(mask_style.unsqueeze(0).long())

        p = [self.scene.mode]

        start_t = time.time()

        z = self.model.netG.encode(mask_m_19[:,1:].to(self.device))
                    
        s_org = self.model.netG.style_encoder(img.unsqueeze(0).to(self.device), mask_19.to(self.device))

        s_swap = self.model.netG.style_encoder(img_style.unsqueeze(0).to(self.device), mask_style_19.to(self.device))

        for part in p:
            s_org[:,part] = s_swap[:,part].clone()  
            
        generated = self.model.netG.decode(z,s_org)

        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))


        result = util.tensor2im(generated[0].squeeze().detach().cpu())      
        result = np.asarray(result, dtype=np.uint8).copy()
        qim = QImage(result.data, result.shape[1], result.shape[0], QImage.Format_RGB888)

        if len(self.result_scene.items())>0: 
            self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(QPixmap.fromImage(qim))


    def make_mask(self, mask, pts, sizes, color):
        if len(pts)>0:
            for idx, pt in enumerate(pts):
                cv2.line(mask,pt['prev'],pt['curr'],(color,color,color),sizes[idx])
        return mask

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                    QDir.currentPath())
            cv2.imwrite(fileName+'.jpg',self.output_img)

    def undo(self):
        self.scene.undo()

    def clear(self):
        self.mask_m = self.mask.copy()
    
        self.scene.reset_items()
        self.scene.reset()
        if type(self.image):
            self.scene.addPixmap(self.image)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    #model = Model(config)
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    vae_model = VAEModel(opt)
    vae_model.eval()

    opt.checkpoints_dir = 'checkpoints/CA2SIS'
    opt.name = 'RGB_model_no_bg'
    model = CA2SISModel(opt)
    model.eval()

    # model = create_model(opt)   
    app = QApplication(sys.argv)
    ex = Ex(vae_model, model, opt)
    sys.exit(app.exec_())

    