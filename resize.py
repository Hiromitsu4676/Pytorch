import cv2
import os
import xml.etree.ElementTree as ET

if __name__=="__main()__":


    img_list=os.listdir('./privateDataset/img')

    for img_name in img_list:
        #画像読み込み
        img=cv2.imread('./privateDataset/img/'+img_name)

        # 画像の大きさを取得
        height, width = img.shape[:2]

        #倍率を算出する
        w_ratio=width/800
        h_ratio=height/600

        #リサイズする
        img_resize=cv2.resize(img,dsize=(800,600))

        
        #画像を保存
        name=img_name[:-4]
        cv2.imwrite('./privateDataset/img_resize/{}.jpg'.format(name),img_resize)

        # XMLを解析 
        tree=ET.parse('./privateDataset/xml/{}.xml'.format(name))

        # XMLを取得 
        root = tree.getroot()
        
        # xmlの書き換え
        for child in root:
            if child.tag == "path":
                child.text="/Users/hiromitsu/Desktop/Python/Pytorch/privateDatasete/img_resize/{}.jpg".format(name)
            if child.tag == "size":
                child.find("width").text = "800"
                child.find("height").text = "600"
            if child.tag == "object":
                child.find("bndbox").find("xmin").text= str(int(int(child.find("bndbox").find("xmin").text)/w_ratio))
                child.find("bndbox").find("ymin").text= str(int(int(child.find("bndbox").find("ymin").text)/h_ratio))
                child.find("bndbox").find("xmax").text= str(int(int(child.find("bndbox").find("xmax").text)/w_ratio))
                child.find("bndbox").find("ymax").text= str(int(int(child.find("bndbox").find("ymax").text)/h_ratio))
        
        #xmlを保存する
        tree.write("./privateDataset/xml_resize/{}.xml".format(name))



            






    

