#XMLファイルをパースして必要な情報を取り出す
import xml.etree.ElementTree as ET 
import numpy as np

class Xml2List(object):
    
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path):
        
        ret = []
        
        xml = ET.parse(xml_path).getroot()
        
        for size in xml.iter("size"):
          
            width = float(size.find("width").text)
            height = float(size.find("height").text)
                
        for obj in xml.iter("object"):
            
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
                
            bndbox = [width, height]
            
            name = obj.find("name").text.lower().strip() 
            bbox = obj.find("bndbox") 
            
            pts = ["xmin", "ymin", "xmax", "ymax"]
            
            for pt in pts:
                
                cur_pixel =  float(bbox.find(pt).text)
                    
                bndbox.append(cur_pixel)
                
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)
            
            ret += [bndbox]
            
        return np.array(ret) # [width, height, xmin, ymin, xamx, ymax, label_idx]