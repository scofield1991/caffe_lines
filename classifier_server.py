# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cStringIO import StringIO
import skimage
import json


# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/opt/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2


class CaffeAlexNet:

    def __init__(self, mode=0, imgs_num=1):

        caffe.set_mode_cpu() if mode == 0 else  caffe.set_mode_gpu()

        self.image_counter = 0
        self.images_array = np.zeros((100, 3, 227, 227))

        #create net
        if imgs_num == 1:
            model_def = '/home/master/caffe_proj/caffe_model_alexnet/caffenet_deploy_2_1_image.prototxt'
        elif imgs_num == 10:
            model_def = '/home/master/caffe_proj/caffe_model_alexnet/caffenet_deploy_2_10_image.prototxt'
        else:
            model_def = '/home/master/caffe_proj/caffe_model_alexnet/caffenet_deploy_2.prototxt'
        model_weights = '/home/master/caffe_proj/caffe_model_alexnet/caffe_model_2_iter_500.caffemodel'

        self.net = caffe.Net(model_def,
                             model_weights,
                             caffe.TEST)
        #get mean image
        mean_blob = caffe_pb2.BlobProto()
        with open('/home/master/caffe_proj/caffe_model_alexnet/mean.binaryproto') as f:
            mean_blob.ParseFromString(f.read())
            mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
                                    (mean_blob.channels, mean_blob.height, mean_blob.width))


        #create transformer for the input called 'data'
        #print (net.blobs['data'].data.shape)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        self.transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        self.transformer.set_mean('data', mean_array)            # subtract the dataset-mean value in each channel
        self.transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

        # set the size of the input (we can skip this if we're happy
        #  with the default; we can also change it later, e.g., for different batch sizes)
        if imgs_num == 1:
            self.net.blobs['data'].reshape(1,  # batch size
                                           3,  # 3-channel (BGR) images
                                           227, 227)  # image size is 227x227
        elif imgs_num == 10:
            self.net.blobs['data'].reshape(10,  # batch size
                                           3,  # 3-channel (BGR) images
                                           227, 227)  # image size is 227x227
        else:
            self.net.blobs['data'].reshape(100,            # batch size
                                       3,         # 3-channel (BGR) images
                                       227, 227)  # image size is 227x227

    def classify_jpg(self, img):

        image = caffe.io.load_image(img)
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)

        # perform classification
        output = self.net.forward()

        # obtain the output probabilities
        output_prob = output['prob'][0]
        top_ind = output_prob.argmax()

        print top_ind

    def classify(self, img):

        #classify image
        #transform it and copy it into the net
        image_64 = img.get('photo').decode('base64')
        image_str = StringIO(image_64)
        im_arr = np.asarray(Image.open(image_str))

        #convert to type suitable for caffe
        img_float = skimage.img_as_float(im_arr).astype(np.float32)
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img_float)

        # perform classification
        output = self.net.forward()

        # obtain the output probabilities
        output_prob = output['prob'][0]
        prob_dict = {'no_line': str(output_prob[0]), 'line': str(output_prob[1])}
        top_ind = output_prob.argmax()

        print prob_dict
        return prob_dict

   

    def json_classifier(self, json_images, num_images):
        images_array = np.zeros((num_images, 3, 227, 227))
        print type(json_images)

        #js_file = open(json_images).read()
        #js_img = json.loads(json_images)
        for num_photo, photo in enumerate(json_images['photo']):
            image_str = StringIO(photo.decode('base64'))
            im_arr = np.asarray(Image.open(image_str))

            # convert to type suitable for caffe
            img_float = skimage.img_as_float(im_arr).astype(np.float32)

            images_array[num_photo] = self.transformer.preprocess('data', img_float)
        self.net.blobs['data'].data[...] = images_array
        output = self.net.forward()

        return output['prob'].tolist()