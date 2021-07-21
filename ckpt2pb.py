import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
def freeze_graph(input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    #output_node_names = "ObjectDetector/detection_output/boxes_conf,ObjectDetector/detection_output/classes"
    output_node_names = "ssd_300_vgg/loc_outputs_all,ssd_300_vgg/pred_outputs_all"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def() # 返回一个序列化的图代表当前的图
    with tf.Session() as sess:

        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        sess.run(tf.global_variables_initializer())
        output_graph_def = graph_util.convert_variables_to_constants( # 模型持久化，将变量值固定
                                                                     sess=sess,
                                                                     input_graph_def=input_graph_def,# 等于:sess.graph_def
                                                                     output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
        input_tensor = tf.get_default_graph().get_tensor_by_name("ssd_preprocessing_train/resize_image/resize/ResizeBilinear:0")
        output_tensor = tf.get_default_graph().get_tensor_by_name("ssd_300_vgg/loc_outputs_all:0")
        pred = {'outs':output_tensor}
        tf.saved_model.simple_save(sess, "./ssd_resnet18_10",inputs={'input_image':input_tensor},outputs={'outs':pred['outs']})
    with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
        f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

    # for op in graph.get_operations():
    # print(op.name, op.values())

import tensorflow as tf
from tensorflow.python.framework import graph_util
'''
def ckpt2pb():
    with tf.Graph().as_default() as graph_old:
        isess = tf.InteractiveSession()

        ckpt_filename = './checkpoint/model.ckpt'
        isess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(ckpt_filename+'.meta', clear_devices=True)
        saver.restore(isess, ckpt_filename)

        constant_graph = graph_util.convert_variables_to_constants(isess, isess.graph_def, ["Cls/fc/biases"])
        constant_graph = graph_util.remove_training_nodes(constant_graph)

        with tf.gfile.GFile('./pb/model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
'''

if __name__ == '__main__':
 # 输入ckpt模型路径
    input_checkpoint='./logs/model.ckpt-53299'
 # 输出pb模型的路径
    out_pb_path="frozen_model.pb"
 # 调用freeze_graph将ckpt转为pb
    freeze_graph(input_checkpoint,out_pb_path)
